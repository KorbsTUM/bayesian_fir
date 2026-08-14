"""
core
====
JAX-native forward model kernels for Bayesian impulse response inference.
"""

from core.parameter_maps import (
    map_to_physical,
    map_to_physical_covariance,
    map_to_parameter,
    unpack_b,
    unpack_a,
)

from core.impulse_response import (
    impulse_response_val,
    calculate_impulse_response,
    impulse_response_jacobian_ad,
    impulse_response_batch,
    gaussian_basis_matrix,
)

from core.prior import (
    PriorConfig,
    generate_prior,
    log_prior,
    extract_nonlinear_prior,
    estimate_t_max,
)

from core.cost import (
    calculate_cost,
    calculate_cost_varpro,
    calculate_cost_val,
    fd_hessian,
)

from core.ftf import (
    calculate_ftf,
)