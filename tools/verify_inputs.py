"""Check released task inputs, embeddings and example backbone files."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1] / "inputs"
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def verify(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Failed input checksum for {path.relative_to(root)}")


def main():
    for task, files in manifest["datasets"].items():
        for name, digest in files.items():
            verify(root / task / name, digest)
    for name, digest in manifest["embeddings"].items():
        verify(root / "embeddings" / name, digest)
    examples = root / "etth2" / "backbone"
    for row in json.loads((examples / "binding.json").read_text(encoding="utf-8")):
        verify(examples / row["released_file"], row["released_sha256"])
    print(f"Verified {len(manifest['datasets'])} tasks and {len(manifest['embeddings'])} text vectors.")


if __name__ == "__main__":
    main()
