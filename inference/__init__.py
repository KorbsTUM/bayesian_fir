"""
inference
=========
MAP optimisation, posterior estimation, MCMC, and model ranking.
"""

from inference.seeds import (
    generate_restart_seeds,
    generate_nonlinear_seeds,
    recommended_restarts,
    recommended_iterations,
)

from inference.optimizer import (
    OptimizerConfig,
    LMState,
    run_one_restart,
    run_all_restarts,
    compute_final_hessian,
)

from inference.posterior import (
    PosteriorResult,
    ImpulseResponse,
    ModelRanking,
    estimate_posterior,
    rank_models,
)

from inference.mcmc import (
    MCMCResult,
    make_proposal_covariance,
    make_log_posterior,
    run_mcmc_scan,
    run_mcmc_python,
    run_mcmc_from_posterior,
)

from inference.sensitivity import (
    make_map_estimator,
    sensitivity_to_physical,
)