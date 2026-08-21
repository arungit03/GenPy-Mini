"""Deep, non-production verification utilities for GenPy."""

from .loss import causal_lm_loss, reference_causal_lm_loss
from .numerical import assert_finite, gradient_audit, is_finite_tensor

__all__ = ["assert_finite", "causal_lm_loss", "gradient_audit", "is_finite_tensor", "reference_causal_lm_loss"]
