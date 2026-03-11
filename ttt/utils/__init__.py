"""Shared runtime utilities used by the in-repo parity JAX stack."""

from .filter_utils import filter_apply_updates, filter_parameters, get_filter_spec, get_mask_fn
from .jax_utils import (
    clone_pytree,
    get_float_dtype_by_name,
    initialize_distibuted,
    master_log,
    maybe_double_remat,
    promote_dtype,
    scan_or_loop,
    scan_remat_chunk,
    set_random_seed,
    tree_rearrange,
    tree_slice,
)

__all__ = [
    "clone_pytree",
    "filter_apply_updates",
    "filter_parameters",
    "get_filter_spec",
    "get_float_dtype_by_name",
    "get_mask_fn",
    "initialize_distibuted",
    "master_log",
    "maybe_double_remat",
    "promote_dtype",
    "scan_or_loop",
    "scan_remat_chunk",
    "set_random_seed",
    "tree_rearrange",
    "tree_slice",
]
