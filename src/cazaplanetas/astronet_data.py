"""Astronet's pre-computed Kepler DR24 views (Shallue & Vanderburg 2018).

The Astronet authors published their training, validation and test sets as
TFRecord files in a Google Drive folder (linked from the exoplanet-ml README):
https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE

This module downloads those 10 files (~145 MB) into ``data/astronet/tfrecord``,
checks them against known SHA-256 sums and converts them once into a single
``.npz`` with plain numpy arrays. TensorFlow is *not* needed: TFRecord framing
and the ``tf.train.Example`` protobuf are simple enough to decode by hand.

Labels follow Astronet's ``label_map``: PC (planet candidate) = 1, AFP
(astrophysical false positive) and NTP (non-transiting phenomenon) = 0.
"""

from __future__ import annotations

import hashlib
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cazaplanetas.env import DATA_DIR

ASTRONET_DIR = DATA_DIR / "astronet"
TFRECORD_DIR = ASTRONET_DIR / "tfrecord"
NPZ_PATH = ASTRONET_DIR / "astronet_dr24.npz"

DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# name -> (Google Drive file id, size in bytes, sha256)
FILES: dict[str, tuple[str, int, str]] = {
    "train-00000-of-00008": ("144fP1v-mlA0I4H2yKnUZK5st_TmeJESY", 14470813,
        "1d7d9004f42df21e4405c0c447041619e62b27052af428b488a7b0803949d9dd"),
    "train-00001-of-00008": ("1ubp6I9C951uggPw2qzTNfxXWqjByvM5v", 14480080,
        "fe8af718082d49684c1157407e8b2e06f8ddb54fc68a0182a0c8526885bd974c"),
    "train-00002-of-00008": ("1q4MAH3GdsA8ZhxHDG_je3jJgvBT5bJ0Q", 14470941,
        "4daa7314280cfc782d8067b06099c02a748a04a58a057bd679d230baf19d8071"),
    "train-00003-of-00008": ("15TpHABm8R8iHz5f09LdltvVs65YipgZ6", 14480082,
        "0eb44faf3e5540f04b6d4562fce3254fd518e2d5c362d4bd0ea3e2b40001ecf9"),
    "train-00004-of-00008": ("1QO98ztq1EntyC8wu0VMrIb8ZSJXBNoqm", 14480033,
        "f714905de8d38d45c46cdad5499d1a8a6a35169d13a7fb5baca57adb1200055d"),
    "train-00005-of-00008": ("1-KHqAwSTq0t0_LEQY-B6QJYyBJ6IG7a_", 14470875,
        "d884ca19a22d504611fb31d61a183b6ec7f597b748083f5454afbd2ddff31d8d"),
    "train-00006-of-00008": ("1KoKCn3_2QKhc5I7gOBt7wKi1kf7fh4nm", 14480051,
        "4d568c5b76f23126bd09a168757879e64b9d208bf9c75dd73794fb9f29646188"),
    "train-00007-of-00008": ("1kIbmzQlDur5On8lZcvYcRQp4_6MrqJi_", 14480040,
        "c29e5053db471b0b230e902ee3e80567eebb8ad747c8ec0b8250ea2c39753311"),
    "val-00000-of-00001": ("1yGrs883G2U-IkXUrM7pxXwgWl3Z2AQFg", 14480066,
        "65a9fc7db3316991a65b5e084ca9cd84f242e99abc4654271805504a30ac1e5b"),
    "test-00000-of-00001": ("12oREWaNDtkFrBd9qteCCkb8MfxCFRsL1", 14480064,
        "db60cb1b0619d832d1c1ea0fef68c8aa515a9619332efd78c6b10e14e4f58ddd"),
}

LABEL_MAP = {"PC": 1, "AFP": 0, "NTP": 0}
SPLITS = ("train", "val", "test")
GLOBAL_LEN, LOCAL_LEN = 2001, 201


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(path: Path, size: int, digest: str) -> bool:
    if not path.exists():
        return False
    if size and path.stat().st_size != size:
        return False
    return not digest or sha256(path) == digest


def download(dest: Path = TFRECORD_DIR, retries: int = 3) -> list[Path]:
    """Download the 10 TFRecord shards (skips files already present and valid)."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (file_id, size, digest) in FILES.items():
        path = dest / name
        for attempt in range(1, retries + 1):
            if _is_valid(path, size, digest):
                break
            tmp = path.with_suffix(".part")
            print(f"  downloading {name} (attempt {attempt})", flush=True)
            try:
                with urllib.request.urlopen(DRIVE_URL.format(id=file_id), timeout=120) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                with open(tmp, "rb") as f:
                    if f.read(1) == b"<":  # Google answered with an HTML page, not the file
                        raise OSError("got an HTML page instead of the TFRecord")
                tmp.replace(path)
            except OSError as e:
                print(f"    failed: {e}", flush=True)
                tmp.unlink(missing_ok=True)
                time.sleep(2 * attempt)
        if not _is_valid(path, size, digest):
            raise RuntimeError(f"could not download a valid copy of {name}")
        paths.append(path)
    return paths


# --- TFRecord + tf.train.Example decoding (no TensorFlow needed) -------------


def read_tfrecord(path: Path):
    """Yield the raw records of a TFRecord file (length-prefixed, CRCs not checked;
    integrity is covered by the SHA-256 of the whole file)."""
    with open(path, "rb") as f:
        while header := f.read(12):
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            f.read(4)  # data crc
            if len(data) != length:
                raise ValueError(f"truncated record in {path}")
            yield data


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, wire_type, value) for a protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i : i + length], i + length
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _decode_feature(buf: bytes):
    for kind, _, payload in _fields(buf):
        if kind == 1:  # BytesList
            return [v for _, _, v in _fields(payload)]
        if kind == 2:  # FloatList (packed or not)
            # Packed (one LEN field) or unpacked (one fixed32 per value): same bytes.
            return np.frombuffer(b"".join(v for _, _, v in _fields(payload)), dtype="<f4")
        if kind == 3:  # Int64List (packed or not)
            out = []
            for _, w, v in _fields(payload):
                if w == 0:
                    out.append(v)
                else:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        out.append(x)
            return np.array([x - (1 << 64) if x >= 1 << 63 else x for x in out], dtype=np.int64)
    return None


def parse_example(record: bytes) -> dict:
    """Decode a serialized ``tf.train.Example`` into {name: numpy array or list of bytes}."""
    out = {}
    for f1, _, features in _fields(record):
        if f1 != 1:
            continue
        for f2, _, entry in _fields(features):
            if f2 != 1:
                continue
            key = value = None
            for f3, _, v in _fields(entry):
                if f3 == 1:
                    key = v.decode()
                elif f3 == 2:
                    value = _decode_feature(v)
            out[key] = value
    return out


# --- Arrays ------------------------------------------------------------------


@dataclass
class Split:
    global_view: np.ndarray  # (n, 2001) float32
    local_view: np.ndarray  # (n, 201) float32
    label: np.ndarray  # (n,) int64, 1 = planet candidate
    kepid: np.ndarray  # (n,) int64
    tce_plnt_num: np.ndarray  # (n,) int64
    av_training_set: np.ndarray  # (n,) str: PC / AFP / NTP

    def __len__(self) -> int:
        return len(self.label)


def convert(tfrecord_dir: Path = TFRECORD_DIR, out: Path = NPZ_PATH) -> Path:
    """Turn the TFRecord shards into one compressed .npz (one-off, a few seconds)."""
    arrays = {}
    for split in SPLITS:
        rows = {k: [] for k in ("global_view", "local_view", "kepid", "tce_plnt_num", "av_training_set")}
        for path in sorted(tfrecord_dir.glob(f"{split}-*-of-*")):
            if path.suffix == ".part":
                continue
            for rec in read_tfrecord(path):
                ex = parse_example(rec)
                rows["global_view"].append(ex["global_view"])
                rows["local_view"].append(ex["local_view"])
                rows["kepid"].append(int(ex["kepid"][0]))
                rows["tce_plnt_num"].append(int(ex["tce_plnt_num"][0]))
                rows["av_training_set"].append(ex["av_training_set"][0].decode())
        if not rows["kepid"]:
            raise RuntimeError(f"no {split} records in {tfrecord_dir}")
        g = np.stack(rows["global_view"]).astype(np.float32)
        loc = np.stack(rows["local_view"]).astype(np.float32)
        assert g.shape[1] == GLOBAL_LEN and loc.shape[1] == LOCAL_LEN, (g.shape, loc.shape)
        av = np.array(rows["av_training_set"])
        arrays[f"{split}_global_view"] = g
        arrays[f"{split}_local_view"] = loc
        arrays[f"{split}_label"] = np.array([LABEL_MAP[a] for a in av], dtype=np.int64)
        arrays[f"{split}_kepid"] = np.array(rows["kepid"], dtype=np.int64)
        arrays[f"{split}_tce_plnt_num"] = np.array(rows["tce_plnt_num"], dtype=np.int64)
        arrays[f"{split}_av_training_set"] = av
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


def load(path: Path = NPZ_PATH) -> dict[str, Split]:
    with np.load(path) as z:
        return {
            s: Split(
                global_view=z[f"{s}_global_view"],
                local_view=z[f"{s}_local_view"],
                label=z[f"{s}_label"],
                kepid=z[f"{s}_kepid"],
                tce_plnt_num=z[f"{s}_tce_plnt_num"],
                av_training_set=z[f"{s}_av_training_set"],
            )
            for s in SPLITS
        }


def prepare() -> dict[str, Split]:
    """Download (if needed), convert (if needed) and load the three splits."""
    download()
    if not NPZ_PATH.exists():
        convert()
    return load()
