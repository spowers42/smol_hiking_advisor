"""Download GGUF model files into the models/ directory.

Usage:
    uv run python tools/download-models.py
    uv run python tools/download-models.py --models-dir /custom/path

Reads model entries from models.ini (llama.cpp preset format).
Skips files that already exist locally.
"""

import argparse
import configparser
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def resolve_quant_file(repo_id: str, quant: str) -> str:
    """Resolve a quant tag (e.g. UD-Q4_K_XL) to the actual .gguf filename."""
    api = HfApi()
    files = api.list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        raise ValueError(f"No .gguf files found in {repo_id}")

    candidates = [f for f in gguf_files if quant.lower() in f.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for c in candidates:
            stem = c.rsplit("/", 1)[-1].replace(".gguf", "")
            if stem.lower() == quant.lower():
                return c
            if stem.lower().endswith(f"-{quant.lower()}") or stem.lower().endswith(quant.lower()):
                return c
        return candidates[0]

    print(
        f"Warning: no file matching '{quant}' in {repo_id}, falling back to {gguf_files[0]}",
        file=sys.stderr,
    )
    return gguf_files[0]


def parse_models_ini(path: str) -> list[dict]:
    config = configparser.ConfigParser()
    config.read(path)
    entries: list[dict] = []
    for section in config.sections():
        if section == "*":
            continue
        hf = config.get(section, "hf", fallback=None)
        if not hf:
            continue
        if ":" in hf:
            repo_id, tag = hf.split(":", 1)
        else:
            repo_id = hf
            tag = ""
        if tag.endswith(".gguf"):
            filename = tag
        elif tag:
            filename = resolve_quant_file(repo_id, tag)
        else:
            filename = ""
        entries.append(
            {
                "alias": section,
                "repo_id": repo_id,
                "filename": filename,
            }
        )
    return entries


def download_entry(entry: dict, models_dir: Path) -> str:
    alias = entry["alias"]
    repo_id = entry["repo_id"]
    filename = entry["filename"]

    if not filename:
        msg = f"No quant tag or filename for '{alias}' in models.ini"
        print(f"[skip] {msg}", file=sys.stderr)
        return ""

    local_path = models_dir / filename
    if local_path.exists():
        print(f"[skip] {alias} -> {local_path} (already exists)")
        return str(local_path)

    print(f"[download] {alias} ({repo_id}:{filename})...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=models_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[done] {alias} -> {downloaded}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models into models/")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Path to models directory (default: <project_root>/models)",
    )
    parser.add_argument(
        "--ini",
        default=None,
        help="Path to models.ini (default: <project_root>/models.ini)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    ini_path = Path(args.ini) if args.ini else root / "models.ini"
    models_dir = Path(args.models_dir) if args.models_dir else root / "models"

    if not ini_path.exists():
        print(f"Error: {ini_path} not found", file=sys.stderr)
        sys.exit(1)

    models_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_models_ini(str(ini_path))

    if not entries:
        print("No model entries found in models.ini")
        sys.exit(0)

    for entry in entries:
        download_entry(entry, models_dir)

    print("\nAll models downloaded to", models_dir)


if __name__ == "__main__":
    main()
