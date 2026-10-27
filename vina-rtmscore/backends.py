"""Coordinate-only GPU RTM client for the independent hybrid v2 experiment."""
from pathlib import Path
import time
import numpy as np
from vina_rtmscore.worker_client import RTMClient
from vina_rtmscore.chemistry import molecule_from_pose
from vina_rtmscore.score_matching import match_scores
from vina_rtmscore.pose_moves import InvalidPoseError
from vina_rtmscore.cached_rtm import canonical_coordinate_molecule, chemistry_signature


class ChemicallyCheckedRelaxer:
    """Reject coordinate-derived stereo changes before candidate archival."""
    def __init__(self, backend, template, index_map):
        self.backend, self.template, self.index_map = backend, template, index_map
        self.expected = chemistry_signature(template)
        self.checked = self.invalid = 0

    def __call__(self, pose):
        value = self.backend(pose)
        value = self._check(value)
        if isinstance(value, InvalidPoseError):
            raise value
        return value

    def _check(self, value):
        if isinstance(value, InvalidPoseError):
            return value
        pose, energy = value
        mol = molecule_from_pose(self.template, pose, self.index_map, 'chemistry_check')
        parsed = canonical_coordinate_molecule(self.template, mol.GetConformer().GetPositions())
        self.checked += 1
        if chemistry_signature(parsed) != self.expected:
            self.invalid += 1
            return InvalidPoseError('coordinate_derived_stereochemistry_changed')
        return value

    def map(self, poses):
        return [self._check(value) for value in self.backend.map(poses)]

    @property
    def stats(self):
        return dict(native=self.backend.stats, chemical_checks=self.checked, invalid_stereochemistry=self.invalid)

    def close(self):
        self.backend.close()


class CoordinateRTMClient(RTMClient):
    def __init__(self, python, rtm_root, directory, pocket, template, index_map,
                 timeout=300., profile=False):
        worker = Path(__file__).resolve().parents[1]/'scripts/rtm_worker.py'
        self.pocket = str(Path(pocket).resolve())
        self.template = template
        self.index_map = index_map
        self.profile = profile
        self.initialization = None
        self.verification = []
        self.timings = dict(calls=0, poses=0, ipc_elapsed_s=0., worker_score_s=0.)
        # Serial order supplied by the frozen Meeko heavy-atom index map.
        self.ordered_serials = [None] * template.GetNumAtoms()
        for atom, serial in index_map.items():
            index = int(atom)-1
            if index < 0 or index >= len(self.ordered_serials):
                raise ValueError('Chemical atom mapping index out of range')
            self.ordered_serials[index] = int(serial)
        if any(x is None for x in self.ordered_serials) or len(set(self.ordered_serials)) != len(self.ordered_serials):
            raise ValueError('Incomplete chemical atom mapping')
        super().__init__(python, worker, rtm_root, directory, timeout)

    def __call__(self, states):
        from rdkit import Chem
        if not states:
            return np.empty(0)
        if self.initialization is None:
            sdf = self.directory/'fixed_chemical_template.sdf'
            with Chem.SDWriter(str(sdf)) as writer:
                state = states[0]
                writer.write(molecule_from_pose(self.template, state['pose'], self.index_map, state['id']))
            reply = self.request(dict(op='initialize', pocket=self.pocket, template_sdf=str(sdf.resolve())))
            if 'error' in reply or not reply.get('initialized') or reply.get('device') != 'cuda':
                raise RuntimeError(f'GPU cache initialization failed: {reply}')
            self.initialization = reply
        coordinates = []
        for state in states:
            pose = state['pose']
            positions = dict(zip(pose.serials, pose.coords))
            xyz = np.asarray([positions[s] for s in self.ordered_serials])
            if not np.allclose(xyz, np.round(xyz, 3), atol=1e-10, rtol=0):
                raise ValueError('Expected scored geometry rounded through official PDBQT')
            coordinates.append(xyz.tolist())
        request = dict(op='score', ids=[s['id'] for s in states], coordinates=coordinates,
                       pocket=self.pocket, verify=not self.verification,
                       profile=self.profile and not self.verification)
        start = time.monotonic(); reply = self.request(request)
        self.timings['ipc_elapsed_s'] += time.monotonic()-start
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        result = match_scores(request['ids'], reply['ids'], reply['scores'])
        self.timings['calls'] += 1; self.timings['poses'] += len(states)
        self.timings['worker_score_s'] += float(reply['elapsed_s'])
        if reply.get('verification'):
            evidence = dict(reply['verification'], initialization=self.initialization)
            if reply.get('profile'):
                evidence['profile'] = reply['profile']
            self.verification.append(evidence)
        return result
