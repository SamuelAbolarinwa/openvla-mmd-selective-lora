import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def replace_remote_refs(obj):
    if isinstance(obj, str):
        return obj.replace("openvla/openvla-7b--", "")
    if isinstance(obj, list):
        return [replace_remote_refs(x) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_remote_refs(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / "config.json").read_text())
    cache = Path(os.environ.get("VLA_CACHE_DIR", args.repo_dir / "cache")).resolve()
    data_root = Path(os.environ.get("VLA_DATA_ROOT", args.repo_dir / "data" / "modified_libero_rlds")).resolve()
    overlay = args.run_dir / "base_model_overlay"
    cache.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    hf_cache = cache / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)

    print("Downloading/pinning OpenVLA base snapshot...")
    snapshot = Path(snapshot_download(
        repo_id=cfg["model_id"],
        revision=cfg["model_revision"],
        cache_dir=hf_cache,
    )).resolve()

    print("Downloading LIBERO-Object RLDS...")
    snapshot_download(
        repo_id=cfg["dataset_repo"],
        repo_type="dataset",
        revision=cfg["dataset_revision"],
        allow_patterns=[f'{cfg["dataset_name"]}/*'],
        local_dir=data_root,
    )

    ds_dir = data_root / cfg["dataset_name"]
    if not ds_dir.exists():
        raise FileNotFoundError(ds_dir)

    # Build a local overlay: metadata/custom code copied, immutable weight shards symlinked.
    if overlay.exists():
        shutil.rmtree(overlay)
    overlay.mkdir(parents=True)

    for item in snapshot.iterdir():
        if not item.is_file():
            continue
        dst = overlay / item.name
        if item.suffix == ".safetensors":
            os.symlink(item, dst)
        else:
            shutil.copy2(item, dst)

    openvla_src = args.repo_dir / "third_party" / "openvla"
    hf_src = openvla_src / "prismatic" / "extern" / "hf"
    for name in [
        "configuration_prismatic.py",
        "modeling_prismatic.py",
        "processing_prismatic.py",
    ]:
        shutil.copy2(hf_src / name, overlay / name)

    for jp in overlay.glob("*.json"):
        try:
            obj = json.loads(jp.read_text())
        except Exception:
            continue
        jp.write_text(json.dumps(replace_remote_refs(obj), indent=2))

    index = overlay / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(index)
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        p = overlay / shard
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty model shard: {p}")

    libero_src = args.repo_dir / "third_party" / "LIBERO" / "libero" / "libero"
    libero_cfg_dir = args.run_dir / "libero_config"
    libero_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Avoid LIBERO's interactive first-run configuration prompt.
    import yaml
    paths = {
        "benchmark_root": str(libero_src),
        "bddl_files": str(libero_src / "bddl_files"),
        "init_states": str(libero_src / "init_files"),
        "datasets": str(args.repo_dir / "data" / "libero_runtime"),
        "assets": str(libero_src / "assets"),
    }
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"LIBERO {key} path missing: {paths[key]}")
    (Path(paths["datasets"])).mkdir(parents=True, exist_ok=True)
    (libero_cfg_dir / "config.yaml").write_text(yaml.safe_dump(paths))

    manifest = {
        "base_model_overlay": str(overlay),
        "base_snapshot": str(snapshot),
        "dataset_root": str(data_root),
        "dataset_dir": str(ds_dir),
        "openvla_source": str(openvla_src.resolve()),
        "libero_source": str((args.repo_dir / "third_party" / "LIBERO").resolve()),
        "libero_config_dir": str(libero_cfg_dir.resolve()),
        "model_revision": cfg["model_revision"],
        "dataset_revision": cfg["dataset_revision"],
        "dlimp_commit": cfg["dlimp_commit"],
        "openvla_commit": cfg["openvla_commit"],
        "libero_commit": cfg["libero_commit"],
    }
    out = args.run_dir / "assets.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("ASSET PREPARATION: PASS")


if __name__ == "__main__":
    main()
