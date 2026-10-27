"""Official Vina 1.2.7 local relaxation with a verified radial-table cache.

The isolated native library reconstructs the official model on every load and
caches only coordinate-independent, by-atom lookup tables. No installed Vina
module is changed. A relaxer is stateful; use one instance per worker.
"""
import ctypes
import math
import os
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vina_rtmscore.pose_moves import InvalidPoseError, Pose


def topology_signature(pose):
    """Complete PDBQT identity/chemistry/tree, excluding only atom coordinates."""
    text = pose.text() if isinstance(pose, Pose) else pose
    return tuple(line[:30] + line[54:] if line.startswith(('ATOM  ', 'HETATM')) else line
                 for line in text.splitlines())


def _library(path):
    lib = ctypes.CDLL(str(path))
    lib.fv_error.restype = ctypes.c_char_p
    lib.fv_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                             ctypes.c_int, ctypes.c_int]
    lib.fv_create.restype = ctypes.c_void_p
    lib.fv_destroy.argtypes = [ctypes.c_void_p]
    lib.fv_optimize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.fv_optimize.restype = ctypes.c_char_p
    lib.fv_score.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
    lib.fv_score.restype = ctypes.c_int
    lib.fv_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    return lib


class FastVinaRelaxer:
    def __init__(self, receptor, start, center, size, directory, max_steps=20, cpu=1,
                 seed=20261021):
        if not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError('max_steps must be a nonnegative integer')
        center = np.asarray(center, dtype=float)
        size = np.asarray(size, dtype=float)
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError('Invalid grid box center or size')
        receptor = Path(receptor)
        if not receptor.is_file():
            raise FileNotFoundError(receptor)
        self._signature = topology_signature(start)
        self.max_steps = max_steps
        self._handle = None
        default = Path(__file__).resolve().parents[1] / 'native/libfast_vina.so'
        path = Path(os.environ.get('FAST_VINA_LIBRARY', str(default)))
        if not path.is_file():
            raise RuntimeError('Isolated Vina library missing; run scripts/build_fast_vina.py --boost-prefix PATH')
        self._lib = _library(path)
        vector = ctypes.c_double * 3
        self._handle = self._lib.fv_create(str(receptor).encode(), start.text().encode(),
                                          vector(*center), vector(*size), cpu, seed)
        if not self._handle:
            self._error()

    def _error(self):
        message = self._lib.fv_error().decode()
        if 'outside the grid box' in message:
            raise InvalidPoseError('optimized_pose_outside_vina_grid: ' + message)
        if 'topology' in message or 'chemistry' in message:
            raise ValueError(message)
        raise RuntimeError(message)

    def __call__(self, pose):
        if not self._handle:
            raise RuntimeError('FastVinaRelaxer is closed')
        if topology_signature(pose) != self._signature:
            raise ValueError('Ligand topology or chemistry changed')
        result = self._lib.fv_optimize(self._handle, pose.text().encode(), self.max_steps)
        if result is None:
            self._error()
        returned = Pose.parse(result.decode())
        if set(returned.serials) != set(pose.serials):
            raise ValueError('Local optimizer changed atom identities')
        index = {s: i for i, s in enumerate(returned.serials)}
        if any(t != returned.types[index[s]] for s, t in zip(pose.serials, pose.types)):
            raise ValueError('Local optimizer changed atom types')
        rounded = pose.with_coords([returned.coords[index[s]] for s in pose.serials])
        energy = ctypes.c_double()
        # Score exactly the rounded PDBQT sent to RTMScore, as the old backend
        # does. This reload reuses the same cached table when atom types match.
        if self._lib.fv_score(self._handle, rounded.text().encode(), ctypes.byref(energy)):
            self._error()
        self.last_raw_energy = energy.value
        value = float(np.around(energy.value, decimals=3))
        if not math.isfinite(value):
            raise ValueError('Non-finite Vina energy')
        return rounded, value

    def map(self, poses):
        """Ordered single-state relaxations with per-proposal invalid errors."""
        results = []
        for pose in poses:
            try:
                results.append(self(pose))
            except InvalidPoseError as exc:
                results.append(exc)
        return results

    @property
    def stats(self):
        if not self._handle:
            raise RuntimeError('FastVinaRelaxer is closed')
        values = (ctypes.c_size_t * 3)()
        self._lib.fv_stats(self._handle, values)
        return dict(table_builds=values[0], table_reuses=values[1], ligand_loads=values[2])

    def close(self):
        if getattr(self, '_handle', None):
            self._lib.fv_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class FastVinaBatchRelaxer:
    """Independent persistent CPU states, leased to threads for entire calls.

    ctypes.CDLL releases the GIL while official C++ minimizes each pose. Each
    state owns its receptor/grid/table, so simultaneous tasks cannot overwrite
    another proposal's geometry. Result order follows proposal order.
    """
    def __init__(self, receptor, start, center, size, directory, max_steps=20,
                 cpu=1, seed=20261021, workers=8):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('workers must be a positive integer')
        self._states = []
        self._available = Queue()
        self._pool = None
        try:
            for i in range(workers):
                state = FastVinaRelaxer(receptor, start, center, size, directory,
                                        max_steps=max_steps, cpu=cpu, seed=seed)
                self._states.append(state)
                self._available.put(state)
            self._pool = ThreadPoolExecutor(max_workers=workers)
        except BaseException:
            self.close()
            raise

    def _relax(self, pose):
        state = self._available.get()
        try:
            try:
                return state(pose)
            except InvalidPoseError as exc:
                return exc
        finally:
            self._available.put(state)

    def __call__(self, pose):
        result = self.map([pose])[0]
        if isinstance(result, InvalidPoseError):
            raise result
        return result

    def map(self, poses):
        if self._pool is None:
            raise RuntimeError('FastVinaBatchRelaxer is closed')
        return list(self._pool.map(self._relax, poses))

    @property
    def stats(self):
        return [state.stats for state in self._states]

    def close(self):
        if getattr(self, '_pool', None) is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for state in getattr(self, '_states', []):
            state.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
