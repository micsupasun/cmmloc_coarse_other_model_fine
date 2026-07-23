"""Checkpoint loading helpers with architecture compatibility checks."""

from collections.abc import Mapping
from pathlib import Path

import torch


_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net")


def _torch_load(path):
    """Load tensor-only checkpoints safely when the installed PyTorch supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 2.0 does not expose the weights_only argument.
        return torch.load(path, map_location="cpu")


def read_state_dict(path):
    """Read a state dict and normalize common training wrappers/prefixes."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = _torch_load(checkpoint_path)
    if not isinstance(state, Mapping):
        raise TypeError(
            f"{checkpoint_path} contains {type(state).__name__}, not a state dict."
        )

    for key in _STATE_DICT_KEYS:
        wrapped = state.get(key)
        if isinstance(wrapped, Mapping):
            state = wrapped
            break

    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        preview = ", ".join(non_tensors[:5])
        raise TypeError(
            f"{checkpoint_path} is not a plain tensor state dict; "
            f"non-tensor entries include: {preview}"
        )

    return dict(state)


def load_model_checkpoint(
    model,
    path,
    *,
    model_name,
    allowed_missing_prefixes=("language_encoder.llm_model.",),
    allow_unexpected=False,
):
    """Load a checkpoint only when it is compatible with ``model``.

    CMMLoc checkpoints intentionally omit the frozen T5 weights. Those keys are
    allowed to be missing; every task-specific layer must match by name and shape.
    """
    state = read_state_dict(path)
    expected = model.state_dict()

    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    disallowed_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    shape_mismatches = sorted(
        (
            key,
            tuple(state[key].shape),
            tuple(expected[key].shape),
        )
        for key in set(state).intersection(expected)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    )

    problems = []
    if disallowed_missing:
        problems.append(
            "missing task-specific keys: " + ", ".join(disallowed_missing[:8])
        )
    if unexpected and not allow_unexpected:
        problems.append("unexpected keys: " + ", ".join(unexpected[:8]))
    if shape_mismatches:
        formatted = ", ".join(
            f"{key} {actual} != {wanted}"
            for key, actual, wanted in shape_mismatches[:8]
        )
        problems.append("shape mismatches: " + formatted)

    if problems:
        raise RuntimeError(
            f"{model_name} checkpoint is incompatible with "
            f"{type(model).__name__}: {'; '.join(problems)}. "
            "Use the matching backend instead of loading this .pth into CMMLoc."
        )

    compatible_state = {
        key: value
        for key, value in state.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }
    model.load_state_dict(compatible_state, strict=False)

    return {
        "checkpoint": str(Path(path)),
        "loaded_tensors": len(compatible_state),
        "missing_pretrained_backbone_tensors": len(missing) - len(disallowed_missing),
        "ignored_unexpected_tensors": len(unexpected),
    }
