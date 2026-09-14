# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens for vision-language models."""
from jvlm.model import VLMLensModel, VLMLayout, load_vlm
from jvlm.fitting import fit, jacobian_for_batch, valid_position_mask
from jvlm import readout
from jvlm.vlens import VLens
from jvlm import direct

__all__ = [
    "VLMLensModel", "VLMLayout", "load_vlm",
    "fit", "jacobian_for_batch", "valid_position_mask", "readout",
]


def lens_from_checkpoint(path):
    """Build a JacobianLens from a (possibly in-progress) fit checkpoint."""
    import torch
    from jlens.lens import JacobianLens
    st = torch.load(path, map_location="cpu", weights_only=True)
    n = st["n_done"]
    return JacobianLens(
        jacobians={l: J / n for l, J in st["jacobian_sum"].items()},
        n_prompts=n,
        d_model=next(iter(st["jacobian_sum"].values())).shape[0],
    )
