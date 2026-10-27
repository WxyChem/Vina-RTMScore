"""Hash the shipped official inference source closure."""
import hashlib
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rtm_source_hashes(root):
    root=Path(root)
    paths=sorted((root/'RTMScore').rglob('*.py'))+[root/'example/rtmscore.py']
    return {str(path.relative_to(root)):digest(path) for path in paths}
