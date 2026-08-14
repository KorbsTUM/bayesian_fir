"""
inference/variational.py
=========================
Normalizing-flow variational inference (VI): an alternative to the Laplace
approximation in inference/posterior.py for estimating the posterior over
the full parameter vector b = [n_1, gamma_1, beta_1, ...] of a fixed model
order N.

Where Laplace approximates the posterior as a single Gaussian centred at
the MAP (mode + local curvature), this module fits a normalizing flow via
the evidence lower bound (ELBO), giving a posterior that can capture skew
and other non-Gaussian shape the Laplace approximation cannot.

Why warm-started from Laplace, not from scratch
-------------------------------------------------
For a fixed model order, the Laplace/LM pipeline (inference.posterior.
estimate_posterior) already reliably finds a good mode via multi-start LM
optimisation - that machinery is not being replaced here. Refitting mode
discovery from an uninformed flow would throw that away and risk landing
in a worse basin. Instead:

    1. Run the existing estimate_posterior(...) once, unmodified, to get
       (b_map, Cb_map, Ce).
    2. Initialise the flow's base affine transform to *exactly* reproduce
       N(b_map, Cb_map) (a lower-triangular matrix parameterised so its
       diagonal is always positive, initialised at the Cholesky factor of
       Cb_map), and initialise every coupling layer to the identity
       transform (zero-initialised final conditioner layer).
    3. Train the whole flow (base affine included) via the ELBO.

At step 0 the flow *is* the Laplace posterior exactly (its samples'
empirical mean/covariance reproduce (b_map, Cb_map) to Monte Carlo noise -
verified against inference.posterior.estimate_posterior directly). Note
this does NOT mean the *ELBO* at step 0 equals Laplace's logML: logML is
an analytic second-order (Gauss-Newton) approximation valid near the mode,
whereas the ELBO is a Monte Carlo expectation over the full N(b_map,
Cb_map) - including its tails, where this model's exp() parameterisation
of (tau, sigma) can make the true cost blow up even for a Gaussian that
looks locally sane. Concretely, on a small synthetic test problem the raw
ELBO at step 0 came out far below logML (dominated by rare bad-tail
samples) and only converged back to within ~1% of it after a few thousand
ELBO steps once the flow learned to shrink/reshape away from those tails.
Ce is held fixed at the Laplace-found value during training - only b is
variational - matching how inference.mcmc already fixes Ce during MCMC
(see its module docstring).

Scope and caveats
-------------------
* The flow operates on the full 3N-dim b, not the VarPro-reduced 2N-dim x.
  calculate_cost_varpro's cost plugs in the conditional MAP of n given x
  without the log|Hn| Occam correction, so it is not the true marginal
  p(q|x); building a flow on top of it would silently target the wrong
  density. calculate_cost (and the value-only calculate_cost_val used
  here) has no such gap - it is the literal full joint negative
  log-posterior - so operating in the full b-space sidesteps the issue
  entirely, at a dimensionality cost that is irrelevant for N <= 5.

* Single mode only. Like a single MCMC chain, this backend only
  characterises the posterior mode the Laplace warm start found. If the
  true posterior is multimodal, cross-order model comparison using this
  backend's logML (the ELBO) inherits that limitation exactly as the
  Laplace path's model comparison already inherits reliance on a single
  quadratic approximation. Mixture-of-flows / multi-seed mode coverage is
  a deliberate non-goal of this pass.

References:
    Rezende & Mohamed (2015), 'Variational Inference with Normalizing Flows'.
    Dinh, Sohl-Dickstein & Bengio (2017), 'Density estimation using Real NVP'.
"""

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.example_libraries import optimizers as jax_optimizers
import distrax

from core.cost import calculate_cost, calculate_cost_val
from core.parameter_maps import map_to_physical, map_to_physical_covariance
from core.impulse_response import impulse_response_batch
from core.prior import PriorConfig
from inference.optimizer import OptimizerConfig
from inference.posterior import PosteriorResult, ImpulseResponse, estimate_posterior


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class VIConfig(NamedTuple):
    """
    Static configuration for the normalizing-flow VI backend.

    All fields are Python scalars so they are treated as compile-time
    constants by JAX, mirroring OptimizerConfig's style.

    Attributes
    ----------
    n_coupling_layers   : int    Number of affine coupling layers (default 4).
    hidden_dim          : int    Hidden width of each coupling layer's
                                 conditioner MLP (default 16).
    n_steps             : int    Number of ELBO training steps (default 3000).
    batch_size          : int    Monte Carlo samples per training step (default 32).
    learning_rate       : float  Adam learning rate (default 1e-3).
    n_posterior_samples : int    Samples drawn for the final posterior
                                 summary (b_map, Cb_map, logML, impulse
                                 response uncertainty). Default 2000.
    log_scale_clip      : float  Soft clamp applied to each coupling layer's
                                 log-scale output as log_scale_clip *
                                 tanh(raw), for training stability. The
                                 identity transform (raw=0) is unaffected
                                 by this clamp regardless of its value.
                                 Default 2.0.
    seed                : int    PRNG seed for flow init, training, and
                                 final sampling. Default 0.
    """
    n_coupling_layers   : int   = 4
    hidden_dim          : int   = 16
    n_steps             : int   = 3000
    batch_size          : int   = 32
    learning_rate       : float = 1e-3
    n_posterior_samples : int   = 2000
    log_scale_clip       : float = 2.0
    seed                 : int   = 0


# ---------------------------------------------------------------------------
# Flow construction
# ---------------------------------------------------------------------------

def _init_coupling_layer(key: jnp.ndarray, d_in: int, d_out: int,
                          hidden_dim: int) -> dict:
    """
    Initialise one coupling layer's conditioner MLP (d_in -> hidden -> 2*d_out).

    The final layer is zero-initialised so the layer starts as the
    identity transform (shift=0, log_scale=0), the standard flow-init trick.
    """
    scale = 1.0 / jnp.sqrt(jnp.maximum(d_in, 1))
    W1 = jax.random.normal(key, (d_in, hidden_dim), dtype=jnp.float64) * scale
    b1 = jnp.zeros(hidden_dim, dtype=jnp.float64)
    W2 = jnp.zeros((hidden_dim, 2 * d_out), dtype=jnp.float64)
    b2 = jnp.zeros(2 * d_out, dtype=jnp.float64)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def init_flow_params(key: jnp.ndarray,
                      b_map: jnp.ndarray,
                      Cb_map: jnp.ndarray,
                      vi_cfg: VIConfig) -> dict:
    """
    Initialise flow parameters so the flow starts as exactly N(b_map, Cb_map).

    Parameters
    ----------
    key    : jax.random.PRNGKey
    b_map  : jnp.ndarray, shape (d,)      Laplace MAP (warm-start mean).
    Cb_map : jnp.ndarray, shape (d, d)    Laplace posterior covariance.
    vi_cfg : VIConfig

    Returns
    -------
    params : dict (pytree) with keys:
        'coupling'      : list of per-layer conditioner param dicts.
        'bias'          : (d,)      base affine shift, init = b_map.
        'raw_log_diag'  : (d,)      log of the base affine's Cholesky
                                    diagonal, init = log(diag(chol(Cb_map))).
                                    Exponentiated when building the flow so
                                    the diagonal stays strictly positive
                                    (hence invertible) throughout training,
                                    regardless of gradient steps.
        'raw_offdiag'   : (d, d)    base affine's strictly-lower Cholesky
                                    entries, init = chol(Cb_map) (masked to
                                    strictly-lower when building the flow).
    """
    d = b_map.shape[0]
    L = jnp.linalg.cholesky(Cb_map)

    n_layers = vi_cfg.n_coupling_layers
    split_index = d // 2
    keys = jax.random.split(key, n_layers)

    coupling = []
    for i in range(n_layers):
        swap = (i % 2 == 1)
        d_in  = (d - split_index) if swap else split_index
        d_out = split_index if swap else (d - split_index)
        coupling.append(_init_coupling_layer(keys[i], d_in, d_out, vi_cfg.hidden_dim))

    return {
        'coupling'    : coupling,
        'bias'        : b_map,
        'raw_log_diag': jnp.log(jnp.diag(L)),
        'raw_offdiag' : L,
    }


def _build_flow(params: dict, d: int, log_scale_clip: float) -> distrax.Transformed:
    """Build the distrax.Transformed flow from a flow_params pytree."""
    split_index = d // 2
    layers = []
    for i, lp in enumerate(params['coupling']):
        swap = (i % 2 == 1)
        d_out = split_index if swap else (d - split_index)

        def make_conditioner(lp=lp, d_out=d_out):
            def conditioner(x_in):
                h = jnp.tanh(x_in @ lp['W1'] + lp['b1'])
                out = h @ lp['W2'] + lp['b2']
                return out.reshape(d_out, 2)
            return conditioner

        def bijector_fn(p, log_scale_clip=log_scale_clip):
            shift = p[..., 0]
            log_scale = log_scale_clip * jnp.tanh(p[..., 1])
            return distrax.Block(distrax.ScalarAffine(shift=shift, log_scale=log_scale), ndims=1)

        layers.append(distrax.SplitCoupling(
            split_index=split_index, event_ndims=1,
            conditioner=make_conditioner(), bijector=bijector_fn, swap=swap))

    # Base affine: L = strictly-lower(raw_offdiag) + diag(exp(raw_log_diag)),
    # guaranteed lower-triangular with strictly positive diagonal (hence
    # invertible) for any value of the trainable parameters.
    L = jnp.tril(params['raw_offdiag'], k=-1) + jnp.diag(jnp.exp(params['raw_log_diag']))
    base_affine = distrax.Chain([
        distrax.Block(distrax.Shift(params['bias']), ndims=1),
        distrax.TriangularLinear(matrix=L, is_lower=True),
    ])

    # distrax.Chain applies the LAST list element first: putting
    # base_affine first in the list means it is applied last, i.e.
    # forward(z) = base_affine(coupling_layers(z)) - coupling layers act
    # in the base-affine-whitened space, base_affine carries the Laplace
    # warm start.
    bijector = distrax.Chain([base_affine] + layers)
    base = distrax.MultivariateNormalDiag(loc=jnp.zeros(d, dtype=jnp.float64),
                                           scale_diag=jnp.ones(d, dtype=jnp.float64))
    return distrax.Transformed(base, bijector)


# ---------------------------------------------------------------------------
# ELBO training
# ---------------------------------------------------------------------------

def _make_elbo_loss(d: int, Ce: float, signals: dict, bp: jnp.ndarray,
                     Cp: jnp.ndarray, T_c: float, prior_cfg: PriorConfig,
                     vi_cfg: VIConfig):
    """
    Build the (params, key) -> scalar negative-ELBO loss function.

    loss(params) = -mean_over_batch[ -J(b) - log_q(b) ]
                 = mean_over_batch[ J(b) + log_q(b) ]

    where b, log_q are drawn via the flow's reparameterized sampler
    (distrax handles the log-det-Jacobian bookkeeping), and J is the
    negative log-posterior from calculate_cost_val (plain differentiable
    JAX - jax.grad backpropagates through flow params -> b -> J directly,
    no custom_vjp needed since nothing here routes through the LM
    optimizer's lax.while_loop).
    """
    def loss_fn(params, key):
        flow = _build_flow(params, d, vi_cfg.log_scale_clip)
        b_batch, log_q = flow.sample_and_log_prob(seed=key, sample_shape=(vi_cfg.batch_size,))

        def neg_log_p(b):
            return calculate_cost_val(signals, Ce, b, bp, Cp, T_c, prior_cfg)

        J_batch = vmap(neg_log_p)(b_batch)
        return jnp.mean(J_batch + log_q)

    return loss_fn


def train_flow(b_map: jnp.ndarray, Cb_map: jnp.ndarray, Ce: float,
                signals: dict, bp: jnp.ndarray, Cp: jnp.ndarray, T_c: float,
                prior_cfg: PriorConfig, vi_cfg: VIConfig) -> dict:
    """
    Train the flow via reparameterized-gradient ELBO maximisation.

    Returns
    -------
    params : dict   Trained flow parameters (same pytree shape as
                    init_flow_params's output).
    """
    d = b_map.shape[0]
    key = jax.random.PRNGKey(vi_cfg.seed)
    key_init, key_train = jax.random.split(key)

    params0 = init_flow_params(key_init, b_map, Cb_map, vi_cfg)
    loss_fn = _make_elbo_loss(d, Ce, signals, bp, Cp, T_c, prior_cfg, vi_cfg)

    opt_init, opt_update, opt_get_params = jax_optimizers.adam(vi_cfg.learning_rate)
    opt_state0 = opt_init(params0)

    @jit
    def step(carry, _):
        opt_state, key = carry
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(opt_get_params(opt_state), subkey)
        opt_state = opt_update(0, grads, opt_state)
        return (opt_state, key), loss

    (opt_state_final, _), loss_history = jax.lax.scan(
        step, (opt_state0, key_train), None, length=vi_cfg.n_steps)

    return opt_get_params(opt_state_final), loss_history


# ---------------------------------------------------------------------------
# Top-level entry point (mirrors inference.posterior.estimate_posterior)
# ---------------------------------------------------------------------------

def estimate_posterior_vi(signals    : dict,
                           bp         : jnp.ndarray,
                           Cp         : jnp.ndarray,
                           T_c        : float,
                           prior_cfg  : PriorConfig,
                           opt_cfg    : OptimizerConfig,
                           vi_cfg     : VIConfig,
                           N          : int,
                           names      : list,
                           Ce0        : Optional[float] = None,
                           n_eval_pts : int = 500) -> PosteriorResult:
    """
    Estimate the posterior for a model of order N via normalizing-flow VI,
    warm-started from the Laplace MAP/covariance. Drop-in alternative to
    inference.posterior.estimate_posterior at the same call site - returns
    the same PosteriorResult type, with posterior_samples populated and
    method='vi'.

    See the module docstring for the warm-start rationale and the
    single-mode / full-b-space scope caveats.

    Parameters
    ----------
    signals, bp, Cp, T_c, prior_cfg, opt_cfg, N, names, Ce0, n_eval_pts
        Same as inference.posterior.estimate_posterior - opt_cfg/Ce0 are
        forwarded unchanged to the Laplace warm-start pass.
    vi_cfg : VIConfig
        Flow architecture / training configuration.

    Returns
    -------
    result : PosteriorResult
    """
    print(f"\nEstimating posterior for N={N} delays (normalizing-flow VI):")

    laplace = estimate_posterior(
        signals, bp, Cp, T_c, prior_cfg, opt_cfg, N, names, Ce0, n_eval_pts)

    print(f"  Warm start (Laplace) : logML={laplace.logML:.4f}  Ce={laplace.Ce:.3e}")

    b_map0  = jnp.asarray(laplace.b_map)
    Cb_map0 = jnp.asarray(laplace.Cb_map)
    Ce      = laplace.Ce

    trained_params, loss_history = train_flow(
        b_map0, Cb_map0, Ce, signals, bp, Cp, T_c, prior_cfg, vi_cfg)

    print(f"  Trained {vi_cfg.n_steps} ELBO steps  |  "
          f"ELBO: {-float(loss_history[0]):.4f} -> {-float(loss_history[-1]):.4f}")

    # ------------------------------------------------------------------
    # Draw the final posterior sample set
    # ------------------------------------------------------------------
    d = b_map0.shape[0]
    key_sample = jax.random.PRNGKey(vi_cfg.seed + 1)
    flow = _build_flow(trained_params, d, vi_cfg.log_scale_clip)
    b_samples, log_q = flow.sample_and_log_prob(
        seed=key_sample, sample_shape=(vi_cfg.n_posterior_samples,))

    J_samples = vmap(lambda b: calculate_cost_val(signals, Ce, b, bp, Cp, T_c, prior_cfg))(b_samples)
    elbo = float(jnp.mean(-J_samples - log_q))

    b_map  = jnp.mean(b_samples, axis=0)
    Cb_map = jnp.cov(b_samples.T)
    Cb_map = (Cb_map + Cb_map.T) / 2.0

    # ------------------------------------------------------------------
    # Model scoring - logML is the ELBO; logBFL/logOF kept for
    # ModelRanking/print_table compatibility (see module docstring: not a
    # literal Occam factor for this backend, just the same reported
    # quantity relation as Laplace uses).
    # ------------------------------------------------------------------
    J_MAP, _, _, _, J_like_full, _ = calculate_cost(
        signals, Ce, b_map, bp, Cp, T_c, prior_cfg)
    logML  = elbo
    logBFL = -float(J_like_full)
    logOF  = logML - logBFL

    # ------------------------------------------------------------------
    # Physical-space point estimate and covariance (linear propagation of
    # the sample covariance - a Gaussian summary of a possibly
    # non-Gaussian posterior, same caveat as elsewhere in this codebase).
    # ------------------------------------------------------------------
    a_map, _, _ = map_to_physical(b_map, T_c)
    Ca_map      = map_to_physical_covariance(b_map, T_c, Cb_map)

    # ------------------------------------------------------------------
    # Impulse response: Monte Carlo mean/variance over the flow's own
    # posterior samples via impulse_response_batch, rather than Laplace's
    # covariance-linearisation - a natural, better-motivated alternative
    # enabled by having real samples.
    # ------------------------------------------------------------------
    sig_coarse = signals['coarse']
    t_h_nd     = sig_coarse['t_h'] / T_c
    t_eval_nd  = jnp.linspace(0.0, float(t_h_nd[-1]), n_eval_pts)

    h_samples = impulse_response_batch(b_samples, t_eval_nd, T_c)   # (n_post, T)
    h_val = jnp.mean(h_samples, axis=0)
    h_var = jnp.var(h_samples, axis=0)

    t_eval_phys = t_eval_nd * T_c
    h = ImpulseResponse(time=t_eval_phys, val=h_val, var=h_var)

    return PosteriorResult(
        b_map  = b_map,
        Cb_map = Cb_map,
        a_map  = a_map,
        Ca_map = Ca_map,
        Ce     = float(Ce),
        h      = h,
        logML  = logML,
        logBFL = logBFL,
        logOF  = logOF,
        N      = N,
        names  = names,
        posterior_samples = b_samples.T,
        method = 'vi',
    )
