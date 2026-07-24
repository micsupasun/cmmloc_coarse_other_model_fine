"""Isolated MNCL fine-stage worker for CMMLoc-produced retrievals."""

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MNCL_FINE_SHA256 = (
    "0a1727faf5108518a83ec182ced6b0e6594f8190267c1533c22c525ea0c62dd4"
)
OFFICIAL_MNCL_POINTNET_SHA256 = (
    "662b00428a6b34f5053382d07bed2ff99897d3653528aca389528069905fc9a2"
)
OFFICIAL_MNCL_COMMIT = "11ea10e1658b38e53b2127f4ee55f9d4236d9f50"
ALLOWED_MISSING_PREFIXES = (
    "language_encoder.llm_model.",
    # The current official source constructs MSG, but its fine forward discards
    # MSG outputs and the released fine checkpoint predates those parameters.
    "language_encoder.MSG.",
)
ALLOWED_LEGACY_CHECKPOINT_PREFIXES = (
    "object_encoder.linear_esa_object.",
    "language_encoder.attention.",
    "language_encoder.linear_text.",
    "language_encoder.linear_q_text.",
    "language_encoder.linear_k_text.",
    "language_encoder.linear_v_text.",
    "language_encoder.gpool.",
    "language_encoder.toare.",
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the official MNCL fine architecture on saved CMMLoc retrievals"
    )
    parser.add_argument("--mncl_root", type=Path, required=True)
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--retrievals_path", type=Path, required=True)
    parser.add_argument("--fine_checkpoint", type=Path, required=True)
    parser.add_argument("--t5_path", dest="hungging_model", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--threshs", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument(
        "--use_features",
        nargs="+",
        default=["class", "color", "position", "num"],
    )
    parser.add_argument("--use_test_set", action="store_true")
    parser.add_argument("--no_pc_augment_fine", action="store_true")
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)
    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", default="all")
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--fine_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--fine_intra_module_num_layers", type=int, default=1)
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_official_checkpoint(path, expected_hash, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} does not match the official MNCL file. "
            f"Expected SHA-256 {expected_hash}, got {actual_hash}: {path}"
        )


def _verify_official_source(path):
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        tracked_changes = subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--short",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not verify MNCL source checkout at {path}: {error}"
        ) from error
    if revision != OFFICIAL_MNCL_COMMIT or tracked_changes:
        raise RuntimeError(
            "MNCL source is not the verified clean official revision. "
            f"Expected {OFFICIAL_MNCL_COMMIT}, got {revision}; "
            f"tracked changes: {tracked_changes or 'none'}. Run "
            "scripts/setup_mncl.ps1 with a clean third_party/MNCL directory."
        )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_tensor_state_dict(torch, path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    state = torch.load(path, **kwargs)
    if not isinstance(state, dict):
        raise TypeError(f"MNCL fine checkpoint is not a state dict: {type(state)}")
    for wrapper in ("state_dict", "model_state_dict", "model", "net"):
        if isinstance(state.get(wrapper), dict):
            state = state[wrapper]
            break
    if state and all(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    non_tensors = [
        key
        for key, value in state.items()
        if not torch.is_tensor(value)
    ]
    if non_tensors:
        raise TypeError(
            "MNCL checkpoint contains non-tensor state entries: "
            + ", ".join(non_tensors[:8])
        )
    return dict(state)


def _load_official_fine_checkpoint(torch, model, path):
    """Load every active MNCL task tensor and reject silent mismatches."""
    state = _load_tensor_state_dict(torch, path)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    disallowed_missing = [
        key
        for key in missing
        if not any(
            key.startswith(prefix)
            for prefix in ALLOWED_MISSING_PREFIXES
        )
    ]
    disallowed_unexpected = [
        key
        for key in unexpected
        if not any(
            key.startswith(prefix)
            for prefix in ALLOWED_LEGACY_CHECKPOINT_PREFIXES
        )
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
            "missing active keys: " + ", ".join(disallowed_missing[:8])
        )
    if disallowed_unexpected:
        problems.append(
            "unknown legacy keys: "
            + ", ".join(disallowed_unexpected[:8])
        )
    if shape_mismatches:
        problems.append(
            "shape mismatches: "
            + ", ".join(
                f"{key} {actual} != {wanted}"
                for key, actual, wanted in shape_mismatches[:8]
            )
        )
    if problems:
        raise RuntimeError(
            "Official MNCL fine checkpoint/source compatibility failed: "
            + "; ".join(problems)
        )

    compatible = {
        key: value
        for key, value in state.items()
        if key in expected
        and tuple(value.shape) == tuple(expected[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    if not compatible:
        raise RuntimeError(
            "No MNCL checkpoint tensors matched the official model."
        )
    return {
        "loaded_tensors": len(compatible),
        "allowed_missing_tensors": missing,
        "allowed_legacy_tensors": unexpected,
        "shape_mismatches": [],
    }


def _seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _construct_official_model(torch, CrossMatch, known_classes, known_colors, args):
    """Construct MNCL while handling its trusted whole-object PointNet checkpoint.

    MNCL was released for PyTorch 1.11 and its ObjectEncoder calls torch.load on
    a full pickled PointNet module. PyTorch 2.6 changed the default to
    weights_only=True, so we restore the legacy behavior only for construction
    and only after verifying the official file's SHA-256.
    """
    pointnet_path = PROJECT_ROOT / "checkpoints" / "pointnet_acc0.86_lr1_p256_model.pth"
    _verify_official_checkpoint(
        pointnet_path, OFFICIAL_MNCL_POINTNET_SHA256, "Official MNCL PointNet"
    )

    original_torch_load = torch.load
    supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters

    if supports_weights_only:
        def trusted_torch_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = trusted_torch_load
    try:
        return CrossMatch(known_classes, known_colors, args)
    finally:
        torch.load = original_torch_load


def _validate_queries(dataset, retrieval_payload):
    retrievals = retrieval_payload.get("retrievals")
    fingerprints = retrieval_payload.get("query_fingerprints")
    if not isinstance(retrievals, list) or not isinstance(fingerprints, list):
        raise ValueError("Invalid retrieval artifact: missing retrievals/fingerprints.")
    if len(dataset.all_poses) != len(retrievals) or len(retrievals) != len(fingerprints):
        raise ValueError(
            "CMMLoc and MNCL produced different query counts: "
            f"MNCL={len(dataset.all_poses)}, artifact={len(retrievals)}."
        )

    for index, (pose, fingerprint) in enumerate(
        zip(dataset.all_poses, fingerprints)
    ):
        same_identity = (
            pose.scene_name == fingerprint["scene_name"]
            and pose.cell_id == fingerprint["cell_id"]
            and np.allclose(pose.pose_w, fingerprint["pose_w"])
        )
        if not same_identity:
            raise ValueError(
                "CMMLoc and MNCL query ordering differs at index "
                f"{index}; refusing an invalid comparison."
            )

    width = max(retrieval_payload["top_k"])
    if any(len(row) != width for row in retrievals):
        raise ValueError(f"Every retrieval row must contain exactly {width} cells.")
    known_cell_ids = {cell.id for cell in dataset.all_cells}
    unknown = sorted(
        {
            cell_id
            for row in retrievals
            for cell_id in row
            if cell_id not in known_cell_ids
        }
    )
    if unknown:
        raise ValueError(
            "CMMLoc retrievals contain cells missing from MNCL's dataset: "
            + ", ".join(unknown[:8])
        )
    return retrievals


def _run_fine(
    torch,
    tqdm,
    model,
    retrievals,
    dataset,
    transform,
    args,
    Kitti360TopKDataset,
    calc_sample_accuracies,
):
    model.eval()
    dataset_topk = Kitti360TopKDataset(
        dataset.all_poses,
        dataset.all_cells,
        retrievals,
        transform,
        args,
    )
    offsets = []
    cell_ids = []
    poses_w = []
    started = time.time()

    with torch.no_grad():
        for sample in tqdm.tqdm(dataset_topk, total=len(dataset_topk)):
            output = model(
                sample["objects"],
                sample["texts"],
                sample["object_points"],
            )
            offsets.append(output.detach().cpu().numpy())
            cell_ids.append([cell.id for cell in sample["cells"]])
            poses_w.append(sample["poses"][0].pose_w)

    print(
        f"Ran MNCL fine matching for {len(dataset_topk)} queries "
        f"in {time.time() - started:0.2f}s."
    )
    all_cells_dict = {cell.id: cell for cell in dataset.all_cells}
    accuracies = {
        k: {threshold: [] for threshold in args.threshs}
        for k in args.top_k
    }

    for index, retrieved_ids in enumerate(tqdm.tqdm(retrievals)):
        pose = dataset.all_poses[index]
        if retrieved_ids != cell_ids[index]:
            raise RuntimeError(f"MNCL cell order changed at query {index}.")
        if not np.allclose(pose.pose_w, poses_w[index]):
            raise RuntimeError(f"MNCL pose order changed at query {index}.")
        top_cells = [all_cells_dict[cell_id] for cell_id in retrieved_ids]
        sample_accuracies = calc_sample_accuracies(
            pose,
            top_cells,
            np.asarray(offsets[index]),
            args.top_k,
            args.threshs,
        )
        for k in args.top_k:
            for threshold in args.threshs:
                accuracies[k][threshold].append(
                    sample_accuracies[k][threshold]
                )

    return {
        k: {
            threshold: float(np.mean(values))
            for threshold, values in threshold_values.items()
        }
        for k, threshold_values in accuracies.items()
    }


def main(argv=None):
    args = _parse_args(argv)
    args.mncl_root = args.mncl_root.resolve()
    args.base_path = args.base_path.resolve()
    args.retrievals_path = args.retrievals_path.resolve()
    args.fine_checkpoint = args.fine_checkpoint.resolve()
    args.output_path = args.output_path.resolve()

    required_source = args.mncl_root / "models" / "cross_matcher.py"
    if not required_source.is_file():
        raise FileNotFoundError(f"Official MNCL source not found: {required_source}")
    _verify_official_source(args.mncl_root)
    _verify_official_checkpoint(
        args.fine_checkpoint,
        OFFICIAL_MNCL_FINE_SHA256,
        "Official MNCL fine checkpoint",
    )

    # MNCL and CMMLoc both use a top-level package named ``models``. Keeping
    # this worker in a separate process and putting MNCL first prevents imports
    # from silently mixing the two architectures.
    sys.path.insert(0, str(args.mncl_root))
    os.chdir(PROJECT_ROOT)

    import torch
    import torch_geometric.transforms as T
    import tqdm

    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti
    from dataloading.kitti360pose.eval import Kitti360TopKDataset
    from datapreparation.kitti360pose.utils import (
        COLOR_NAMES as COLOR_NAMES_K360,
        KNOWN_CLASS,
        SCENE_NAMES_TEST,
        SCENE_NAMES_VAL,
    )
    from evaluation.utils import calc_sample_accuracies, print_accuracies
    from models.cross_matcher import CrossMatch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError(
            "The official MNCL implementation contains CUDA-only operations. "
            "Run this comparison on the university GPU machine."
        )

    with args.retrievals_path.open("r", encoding="utf-8") as handle:
        retrieval_payload = json.load(handle)
    expected_split = "test" if args.use_test_set else "validation"
    if retrieval_payload.get("split") != expected_split:
        raise ValueError(
            f"Retrieval split is {retrieval_payload.get('split')}, "
            f"but MNCL worker requested {expected_split}."
        )
    if retrieval_payload.get("top_k") != args.top_k:
        raise ValueError("MNCL top_k does not match the CMMLoc retrieval artifact.")

    transform = (
        T.FixedPoints(args.pointnet_numpoints)
        if args.no_pc_augment_fine
        else T.Compose(
            [T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()]
        )
    )
    scenes = SCENE_NAMES_TEST if args.use_test_set else SCENE_NAMES_VAL
    dataset = Kitti360CoarseDatasetMulti(
        str(args.base_path),
        scenes,
        transform,
        shuffle_hints=False,
        flip_poses=False,
    )
    retrievals = _validate_queries(dataset, retrieval_payload)

    model = _construct_official_model(
        torch, CrossMatch, KNOWN_CLASS, COLOR_NAMES_K360, args
    )
    load_report = _load_official_fine_checkpoint(
        torch,
        model,
        args.fine_checkpoint,
    )
    model.to(device)

    # Match the point subsets used by the other fine backends. This reset is
    # deliberately after construction/loading because those steps may consume
    # random numbers even though evaluation itself is deterministic.
    _seed_everything(torch, args.eval_seed)
    accuracies = _run_fine(
        torch,
        tqdm,
        model,
        retrievals,
        dataset,
        transform,
        args,
        Kitti360TopKDataset,
        calc_sample_accuracies,
    )
    print_accuracies(accuracies, "Fine (MNCL)")

    result = {
        "backend": "official_mncl",
        "source": "https://github.com/dqliua/MNCL",
        "source_commit": OFFICIAL_MNCL_COMMIT,
        "checkpoint": str(args.fine_checkpoint),
        "checkpoint_sha256": OFFICIAL_MNCL_FINE_SHA256,
        "text_backbone": args.hungging_model,
        "eval_seed": args.eval_seed,
        "load_report": load_report,
        "accuracies": accuracies,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, indent=2, ensure_ascii=False)
    print(f"Saved MNCL result: {args.output_path}")
    return result


if __name__ == "__main__":
    main()
