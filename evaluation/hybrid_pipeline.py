"""Use one CMMLoc coarse stage with multiple fine-stage architectures.

The coarse retrievals are computed once and saved as a small, architecture-
neutral JSON artifact. CMMLoc-compatible fine checkpoints run in this process;
MNCL runs in an isolated subprocess so its top-level ``models`` package cannot
collide with CMMLoc's package.
"""

import copy
import hashlib
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
from models.fine.cross_matcher import CrossMatch as CMMLocCrossMatch
from models.my_model.cross_matcher import CrossMatch as MyModelCrossMatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "k360_30-10_scG_pd10_pc4_spY_all"
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "k360_30-10_scG_pd10_pc4_spY_all"
)
SUPPORTED_FINE_MODELS = ("CMMLoc", "MNCL", "my_model")
FINE_MODEL_BACKENDS = {
    "CMMLoc": ("official_cmmloc", CMMLocCrossMatch),
    "my_model": ("cmmloc_mnclv4", MyModelCrossMatch),
}
MY_MODEL_SOURCE_HASHES = {
    "models/fine/cross_matcher.py": (
        "23bcc6e87a48aeb6ff8014e34f6ada1852b78410dd1bc2a18894bdb422c49846"
    ),
    "models/fine/language_encoder.py": (
        "3a5cff499e9c29b842fa30492c517c9ba4d6525545a38fea8d1e14b7d74aa3b3"
    ),
    "models/fine/object_encoder.py": (
        "21b2b240343948a48798342c8ea61f2c81015b24730a09f471fe384870dd7a85"
    ),
}
EXPECTED_CHECKPOINT_SHA256 = {
    "CMMLoc/coarse.pth": (
        "5e14e158c3de1fc046d9b970ef1d06c6d4a98d55a1cfdd09f6d26dfc23076f85"
    ),
    "CMMLoc/fine.pth": (
        "720623e7e25866b0e552b83080202bd3ec855672e0bc83cd962c173506cd648a"
    ),
    "MNCL/fine.pth": (
        "0a1727faf5108518a83ec182ced6b0e6594f8190267c1533c22c525ea0c62dd4"
    ),
    "my_model/fine.pth": (
        "acea2f8fbe58aae256d942606dfd269fa9c3b486849b64edd24cf8720c1fbe1e"
    ),
}


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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint_hash(path, artifact_name):
    actual = _sha256(path)
    expected = EXPECTED_CHECKPOINT_SHA256[artifact_name]
    if actual != expected:
        raise RuntimeError(
            f"{artifact_name} is not the checkpoint audited for this "
            f"comparison. Expected SHA-256 {expected}, got {actual}: {path}"
        )
    return actual


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
        "--mncl_t5_path",
        type=str,
        default="google/flan-t5-large",
        help=(
            "MNCL Flan-T5-large directory or Hugging Face model id. This is "
            "separate from --cmmloc_t5_path because the released models use "
            "different text backbones."
        ),
    )
    parser.add_argument(
        "--my_model_t5_path",
        type=str,
        default="google-t5/t5-large",
        help=(
            "T5-large directory/model id for CMMLoc_MNCLv4. This is separate "
            "so the custom model cannot inherit another backend's text "
            "configuration accidentally."
        ),
    )
    parser.add_argument(
        "--my_model_text_max_length",
        type=int,
        default=128,
        help=(
            "Token limit used by CMMLoc_MNCLv4 fine training/evaluation "
            "(the v4 evaluation default is 128)."
        ),
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Reset before every fine backend so point sampling is identical.",
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
        hungging_model="google-t5/t5-large",
        no_pc_augment=True,
        no_pc_augment_fine=True,
        fixed_embedding=True,
    )
    return parser


def _resolve_model_source(value, *, label):
    candidate = Path(value)
    if candidate.exists():
        return str(candidate.resolve())
    if candidate.is_absolute() or value.startswith("."):
        raise FileNotFoundError(
            f"{label} model directory not found: {value}"
        )
    return value


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
    _verify_checkpoint_hash(args.path_coarse, "CMMLoc/coarse.pth")

    pointnet_path = Path(args.pointnet_path).resolve()
    if not pointnet_path.is_file():
        raise FileNotFoundError(
            f"Coarse PointNet checkpoint not found: {pointnet_path}"
        )
    args.pointnet_path = str(pointnet_path)

    args.hungging_model = _resolve_model_source(
        args.hungging_model, label="CMMLoc T5-large"
    )
    args.mncl_t5_path = _resolve_model_source(
        args.mncl_t5_path, label="MNCL Flan-T5-large"
    )
    args.my_model_t5_path = _resolve_model_source(
        args.my_model_t5_path, label="CMMLoc_MNCLv4 T5-large"
    )
    cmmloc_source = args.hungging_model.replace("\\", "/").lower()
    if "flan-t5" in cmmloc_source:
        raise ValueError(
            "Released CMMLoc checkpoints require the original T5-large "
            "backbone, but a Flan-T5 model was passed via "
            f"--cmmloc_t5_path: {args.hungging_model}. Use "
            "google-t5/t5-large (or the matching local t5-large directory)."
        )
    my_model_source = args.my_model_t5_path.replace("\\", "/").lower()
    if "flan-t5" in my_model_source:
        raise ValueError(
            "CMMLoc_MNCLv4 was trained with standard T5-large, but a "
            "Flan-T5 model was passed via --my_model_t5_path: "
            f"{args.my_model_t5_path}."
        )
    if args.my_model_text_max_length <= 0:
        raise ValueError("--my_model_text_max_length must be positive.")

    if not args.coarse_only:
        for model_name in args.fine_models:
            checkpoint = args.checkpoint_root / model_name / "fine.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"{model_name} fine checkpoint not found: {checkpoint}"
                )
            _verify_checkpoint_hash(
                checkpoint,
                f"{model_name}/fine.pth",
            )

        for model_name in set(args.fine_models).intersection(
            FINE_MODEL_BACKENDS
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
        "coarse_checkpoint_sha256": _sha256(args.path_coarse),
        "text_backbone": args.hungging_model,
        "eval_seed": args.eval_seed,
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


def _fine_args_for_model(args, model_name):
    """Return isolated construction/evaluation args for one fine backend."""
    model_args = copy.copy(args)
    model_dir = args.checkpoint_root / model_name
    model_args.prealign_mlp_path = str(model_dir / "prealign_mlp.pth")
    model_args.prealign_color_path = str(
        model_dir / "prealign_color_encoder.pth"
    )
    model_args.prealign_pointnet_path = str(
        model_dir / "prealign_pointnet.pth"
    )
    if model_name == "my_model":
        model_args.hungging_model = args.my_model_t5_path
        model_args.text_max_length = args.my_model_text_max_length
    else:
        model_args.hungging_model = args.hungging_model
    return model_args


def _run_local_fine_models(
    args, device, retrievals, dataloader, transform_fine
):
    selected = [
        model_name
        for model_name in args.fine_models
        if model_name in FINE_MODEL_BACKENDS
    ]
    if not selected:
        return {}

    results = {}
    for model_name in selected:
        backend_name, model_class = FINE_MODEL_BACKENDS[model_name]
        model_args = _fine_args_for_model(args, model_name)
        checkpoint = args.checkpoint_root / model_name / "fine.pth"
        print(
            f"Constructing isolated {model_name} fine backend: "
            f"{backend_name}"
        )
        model = model_class(
            KNOWN_CLASS,
            COLOR_NAMES_K360,
            model_args,
        ).to(device)
        report = load_model_checkpoint(
            model,
            checkpoint,
            model_name=model_name,
        )

        # FixedPoints may sample points lazily. Reset after model construction
        # so every backend receives the same random point subset regardless of
        # initialization work or evaluation order.
        _seed_everything(args.eval_seed)
        accuracies = run_fine(
            model,
            retrievals,
            dataloader,
            model_args,
            transform_fine,
        )
        print_accuracies(accuracies, f"Fine ({model_name})")
        result = {
            "backend": backend_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "text_backbone": model_args.hungging_model,
            "text_max_length": (
                model_args.text_max_length
                if model_name == "my_model"
                else None
            ),
            "eval_seed": args.eval_seed,
            "load_report": report,
            "accuracies": accuracies,
        }
        if model_name == "my_model":
            result["source_lineage"] = "CMMLoc_MNCLv4"
            result["audited_source_sha256"] = MY_MODEL_SOURCE_HASHES
        results[model_name] = result
        _write_json(
            args.output_dir / f"{model_name.lower()}_fine.json",
            result,
        )

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
        args.mncl_t5_path,
        "--output_path",
        str(result_path),
        "--device",
        args.device,
        "--eval_seed",
        str(args.eval_seed),
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
    _seed_everything(args.eval_seed)
    device = _select_device(args.device)
    device_label = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print(f"device: {device} ({device_label})")
    print(f"fine models: {', '.join(args.fine_models)}")
    print(f"CMMLoc text backbone: {args.hungging_model}")
    if "my_model" in args.fine_models and not args.coarse_only:
        print(
            "my_model backend: CMMLoc_MNCLv4; "
            f"text backbone: {args.my_model_t5_path}; "
            f"max length: {args.my_model_text_max_length}"
        )
    if "MNCL" in args.fine_models and not args.coarse_only:
        print(f"MNCL text backbone: {args.mncl_t5_path}")

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
    print(f"CMMLoc coarse checkpoint load: {coarse_load_report}")
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

    fine_results = {}
    if not args.coarse_only:
        fine_results = _run_local_fine_models(
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
        "eval_seed": args.eval_seed,
        "coarse": {
            "model": "CMMLoc",
            "checkpoint": args.path_coarse,
            "checkpoint_sha256": _sha256(args.path_coarse),
            "text_backbone": args.hungging_model,
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
