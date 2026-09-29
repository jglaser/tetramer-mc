"""Validated, frozen subsets of a fixed global super-Fibonacci rotation grid."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import numpy as np


def validate_rotation_indices(indices, rotations):
    if rotations <= 0:
        raise ValueError('rotations must be positive')
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError('rotation indices must be a one-dimensional integer array')
    if len(values) == 0:
        raise ValueError('rotation index subset must not be empty')
    if np.any(values < 0) or np.any(values >= rotations):
        raise ValueError('rotation indices must lie in [0, rotations)')
    result = np.sort(values.astype(np.int64))
    if np.any(result[1:] == result[:-1]):
        raise ValueError('rotation indices must be unique')
    return result


def load_rotation_indices(path, rotations):
    return validate_rotation_indices(np.load(path, allow_pickle=False), rotations)


def shard_rotation_indices(rotations, rank, parts, path=None):
    if rotations <= 0 or parts <= 0 or not 0 <= rank < parts:
        raise ValueError('invalid global rotation count or shard specification')
    if path is None:
        # Preserve the exact original no-subset order and arithmetic.
        return np.arange(rank, rotations, parts)
    return load_rotation_indices(path, rotations)[rank::parts]


def freeze_rotation_indices(source, destination, rotations):
    """Validate and archive the exact bytes once; execution uses ascending IDs."""
    raw = Path(source).read_bytes()
    indices = validate_rotation_indices(np.load(io.BytesIO(raw), allow_pickle=False), rotations)
    with Path(destination).open('xb') as stream:
        stream.write(raw)
    return dict(rotation_indices=str(Path(destination).resolve()),
                rotation_indices_source=str(Path(source).resolve()),
                rotation_indices_sha256=hashlib.sha256(raw).hexdigest(),
                selected_rotations=len(indices), rotation_indices_order='ascending',
                rotation_index_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
