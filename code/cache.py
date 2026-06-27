"""Disk cache for Stage-1 VLM responses keyed by sha256(image_bytes + claim_key + model)."""
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"


def _key(image_bytes_list: list, claim_key: str, model: str) -> str:
    h = hashlib.sha256()
    for b in image_bytes_list:
        h.update(b)
    h.update(b"\x00")
    h.update(claim_key.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def get(image_bytes_list: list, claim_key: str, model: str):
    k = _key(image_bytes_list, claim_key, model)
    path = CACHE_DIR / f"{k}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def put(image_bytes_list: list, claim_key: str, model: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = _key(image_bytes_list, claim_key, model)
    path = CACHE_DIR / f"{k}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
