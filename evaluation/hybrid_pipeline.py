"""Use one CMMLoc coarse stage with multiple fine-stage architectures.

The coarse retrievals are computed once and saved as a small, architecture-
neutral JSON artifact. CMMLoc-compatible fine checkpoints run in this process;
MNCL runs in an isolated subprocess so its top-level ``models`` package cannot
collide with CMMLoc's package.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import DataLoader

from dataloading.kitti360pose.cells import (
    Kitti360CoarseDataset,
    Kitti360CoarseDatasetMulti,
)
from datapreparation.kitti360pose.utils import (
    COLOR_NAMES as COLOR_NAMES_K360,
    KNOWN_CLASS,
    SCENE_NAMES_TEST,
    SCENE_NAMES_VAL,
)
from evaluation.args import build_parser
from evaluation.checkpoints import load_model_checkpoint
from evaluation.pipeline import run_coarse, run_fine
from evaluation.utils import print_accuracies
from models.coarse.cell_retrieval import CellRetrievalNetwork
from models.fine.cross_matcher import CrossMatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "k360_30-10_scG_pd10_pc4_spY_all"
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "k360_30-10_scG_pd10_pc4_spY_all"
)
CMMLOC_COMPATIBLE_FINE_MODELS = ("CMMLoc", "my_model")
SUPPORTED_FINE_MODELS = ("CMMLoc", "MNCL", "my_model")


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


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def _build_parser():
    parser = build_parser()
    parser.description = (
        "CMMLoc coarse retrieval followed by CMMLoc, MNCL, and/or custom fine stages"
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory containing CMMLoc/, MNCL/, and my_model/.",
    )
    parser.add_argument(
        "--fine_models",
        nargs="+",
        choices=SUPPORTED_FINE_MODELS,
        default=list(SUPPORTED_FINE_MODELS),
        help="Fine stages to evaluate against the same CMMLoc retrievals.",
    )
    parser.add_argument(
        "--mncl_root",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "MNCL",
        help="Checkout of https://github.com/dqliua/MNCL.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cmmloc_coarse_multi_fine",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. auto selects CUDA when available.",
    )
    parser.add_argument(
        "--pc_augment",
        dest="no_pc_augment",
        action="store_false",
        help="Enable the coarse point-cloud augmentation used during training.",
    )
    parser.add_argument(
        "--pc_augment_fine",
        dest="no_pc_augment_fine",
        action="store_false",
        help="Enable the fine point-cloud augmentation used during training.",
    )
    parser.set_defaults(
        base_path=str(DEFAULT_DATASET),
        pointnet_path=str(
            DEFAULT_CHECKPOINT_ROOT / "CMMLoc" / "prealign_pointnet.pth"
        ),
        hungging_model=str(PROJECT_ROOT / "t5-large"),
        no_pc_augment=True,
        no_pc_augment_fine=True,
        fixed_embedding=True,
    )
    return parser


def _validate_args(args):
    args.base_path = str(Path(args.base_path).resolve())
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.mncl_root = args.mncl_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if not Path(args.base_path).is_dir():
        raise FileNotFoundError(f"KITTI360Pose directory not found: {args.base_path}")
    if not args.checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint root not found: {args.checkpoint_root}"
        )

    coarse_checkpoint = (
        Path(args.path_coarse).resolve()
        if args.path_coarse
        else args.checkpoint_root / "CMMLoc" / "coarse.pth"
    )
    if not coarse_checkpoint.is_file():
        raise FileNotFoundError(
            f"CMMLoc coarse checkpoint not found: {coarse_checkpoint}"
        )
    args.path_coarse = str(coarse_checkpoint)

    pointnet_path = Path(args.pointnet_path).resolve()
    if not pointnet_path.is_file():
        raise FileNotFoundError(
            f"Coarse PointNet checkpoint not found: {pointnet_path}"
        )
    args.pointnet_path = str(pointnet_path)

    t5_candidate = Path(args.hungging_model)
    if t5_candidate.exists():
        args.hungging_model = str(t5_candidate.resolve())
    elif t5_candidate.is_absolute() or args.hungging_model.startswith("."):
        raise FileNotFoundError(
            "T5 model directory not found. Place it in t5-large/ or pass a "
            f"Hugging Face model id with --t5_path: {args.hungging_model}"
        )

    for model_name in args.fine_models:
        checkpoint = args.checkpoint_root / model_name / "fine.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{model_name} fine checkpoint not found: {checkpoint}"
            )

    for model_name in set(args.fine_models).intersection(
        CMMLOC_COMPATIBLE_FINE_MODELS
    ):
        model_dir = args.checkpoint_root / model_name
        for filename in (
            "prealign_mlp.pth",
            "prealign_color_encoder.pth",
            "prealign_pointnet.pth",
        ):
            path = model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{model_name} construction checkpoint not found: {path}"
                )

    if "MNCL" in args.fine_models:
        required_mncl_files = (
            args.mncl_root / "models" / "cross_matcher.py",
            args.mncl_root / "dataloading" / "kitti360pose" / "eval.py",
        )
        missing = [path for path in required_mncl_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MNCL source checkout is missing. Run scripts/setup_mncl.ps1 "
                f"or clone https://github.com/dqliua/MNCL into {args.mncl_root}. "
                f"Missing: {', '.join(map(str, missing))}"
            )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def _select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_dataset(args):
    transform = (
        T.FixedPoints(args.pointnet_numpoints)
        if args.no_pc_augment
        else T.Compose(
            [T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()]
        )
    )
    transform_fine = (
        T.FixedPoints(args.pointnet_numpoints)
        if args.no_pc_augment_fine
        else T.Compose(
            [T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()]
        )
    )
    scenes = SCENE_NAMES_TEST if args.use_test_set else SCENE_NAMES_VAL
    dataset = Kitti360CoarseDatasetMulti(
        args.base_path,
        scenes,
        transform,
        shuffle_hints=False,
        flip_poses=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    return dataloader, transform_fine


def _retrieval_payload(args, dataloader, retrievals, coarse_accuracies):
    return {
        "schema_version": 1,
        "producer": "CMMLoc",
        "split": "test" if args.use_test_set else "validation",
        "base_path": args.base_path,
        "top_k": list(args.top_k),
        "thresholds": list(args.threshs),
        "coarse_checkpoint": args.path_coarse,
        "coarse_accuracies": coarse_accuracies,
        "query_fingerprints": [
            {
                "scene_name": pose.scene_name,
                "cell_id": pose.cell_id,
                "pose_w": np.asarray(pose.pose_w).tolist(),
            }
            for pose in dataloader.dataset.all_poses
        ],
        "retrievals": retrievals,
    }


def _set_prealign_paths(args, model_name):
    model_dir = args.checkpoint_root / model_name
    args.prealign_mlp_path = str(model_dir / "prealign_mlp.pth")
    args.prealign_color_path = str(model_dir / "prealign_color_encoder.pth")
    args.prealign_pointnet_path = str(model_dir / "prealign_pointnet.pth")


def _run_cmmloc_compatible_fine_models(
    args, device, retrievals, dataloader, transform_fine
):
    selected = [
        model_name
        for model_name in args.fine_models
        if model_name in CMMLOC_COMPATIBLE_FINE_MODELS
    ]
    if not selected:
        return {}

    # Both checkpoints have the same CMMLoc fine architecture. Construct once;
    # the compatibility checker guarantees that each task-specific tensor is
    # overwritten before inference.
    _set_prealign_paths(args, selected[0])
    model = CrossMatch(KNOWN_CLASS, COLOR_NAMES_K360, args).to(device)
    results = {}

    for model_name in selected:
        checkpoint = args.checkpoint_root / model_name / "fine.pth"
        report = load_model_checkpoint(
            model,
            checkpoint,
            model_name=model_name,
        )
        accuracies = run_fine(
            model, retrievals, dataloader, args, transform_fine
        )
        print_accuracies(accuracies, f"Fine ({model_name})")
        results[model_name] = {
            "backend": "cmmloc",
            "checkpoint": str(checkpoint),
            "load_report": report,
            "accuracies": accuracies,
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _run_mncl_fine(args, retrievals_path):
    result_path = args.output_dir / "mncl_fine.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "mncl_fine_worker.py"),
        "--mncl_root",
        str(args.mncl_root),
        "--base_path",
        args.base_path,
        "--retrievals_path",
        str(retrievals_path),
        "--fine_checkpoint",
        str(args.checkpoint_root / "MNCL" / "fine.pth"),
        "--t5_path",
        args.hungging_model,
        "--output_path",
        str(result_path),
        "--device",
        args.device,
        "--top_k",
        *map(str, args.top_k),
        "--threshs",
        *map(str, args.threshs),
        "--use_features",
        *args.use_features,
        "--fine_embed_dim",
        str(args.fine_embed_dim),
        "--fine_num_decoder_heads",
        str(args.fine_num_decoder_heads),
        "--fine_num_decoder_layers",
        str(args.fine_num_decoder_layers),
        "--pad_size",
        str(args.pad_size),
        "--num_mentioned",
        str(args.num_mentioned),
        "--pointnet_numpoints",
        str(args.pointnet_numpoints),
        "--pointnet_features",
        str(args.pointnet_features),
        "--fine_intra_module_num_heads",
        str(args.fine_intra_module_num_heads),
        "--fine_intra_module_num_layers",
        str(args.fine_intra_module_num_layers),
    ]
    if args.use_test_set:
        command.append("--use_test_set")
    if args.no_pc_augment_fine:
        command.append("--no_pc_augment_fine")
    if args.fixed_embedding:
        command.append("--fixed_embedding")
    if args.pointnet_freeze:
        command.append("--pointnet_freeze")
    if args.class_embed:
        command.append("--class_embed")
    if args.color_embed:
        command.append("--color_embed")

    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    _seed_everything(42)
    device = _select_device(args.device)
    device_label = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print(f"device: {device} ({device_label})")
    print(f"fine models: {', '.join(args.fine_models)}")

    dataloader, transform_fine = _make_dataset(args)

    model_coarse = CellRetrievalNetwork(
        KNOWN_CLASS, COLOR_NAMES_K360, args
    ).to(device)
    coarse_load_report = load_model_checkpoint(
        model_coarse,
        args.path_coarse,
        model_name="CMMLoc coarse",
        # Released CMMLoc checkpoints contain training-only modules that the
        # official inference code loads with strict=False. Still reject any
        # missing current-model key or tensor-shape mismatch.
        allow_unexpected=True,
    )
    retrievals, coarse_accuracies = run_coarse(
        model_coarse, dataloader, args
    )
    print_accuracies(coarse_accuracies, "Coarse (CMMLoc)")

    retrievals_path = args.output_dir / "cmmloc_retrievals.json"
    _write_json(
        retrievals_path,
        _retrieval_payload(
            args, dataloader, retrievals, coarse_accuracies
        ),
    )
    del model_coarse
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fine_results = _run_cmmloc_compatible_fine_models(
        args, device, retrievals, dataloader, transform_fine
    )
    if "MNCL" in args.fine_models:
        fine_results["MNCL"] = _run_mncl_fine(args, retrievals_path)

    summary = {
        "schema_version": 1,
        "split": "test" if args.use_test_set else "validation",
        "query_count": len(retrievals),
        "top_k": list(args.top_k),
        "thresholds": list(args.threshs),
        "coarse": {
            "model": "CMMLoc",
            "checkpoint": args.path_coarse,
            "load_report": coarse_load_report,
            "accuracies": coarse_accuracies,
            "retrievals_path": str(retrievals_path),
        },
        "fine": fine_results,
    }
    summary_path = args.output_dir / "comparison.json"
    _write_json(summary_path, summary)
    print(f"Saved comparison: {summary_path}")
    return summary


if __name__ == "__main__":
    main()
