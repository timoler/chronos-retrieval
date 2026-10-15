"""Restore only recorded LF/CRLF differences, then check exact execution hashes."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', required=True, type=Path)
args = parser.parse_args()
root = args.source_root.resolve()
manifest = json.loads((Path(__file__).parent / 'after_io_retry/CODE_MANIFEST.json').read_text(encoding='utf-8'))
digest = lambda value: hashlib.sha256(value).hexdigest()
pending = []
for relative, expected in manifest.items():
    path = (root / relative).resolve()
    path.relative_to(root)
    original = path.read_bytes()
    if digest(original) == expected:
        continue
    lf = original.replace(b'\r\n', b'\n')
    matches = [value for value in [lf, lf.replace(b'\n', b'\r\n')] if digest(value) == expected]
    if not matches:
        raise RuntimeError('Source content differs beyond line endings: ' + relative)
    pending.append((path, original, matches[0]))
for path, original, restored in pending:
    if path.read_bytes() != original:
        raise RuntimeError('Source changed during verification: ' + str(path.relative_to(root)))
    path.write_bytes(restored)
assert all(digest((root / relative).read_bytes()) == expected for relative, expected in manifest.items())
print(json.dumps({'status': 'PASS', 'files': len(manifest), 'line_endings_restored': len(pending)}))
