"""Exact RTMScore coordinate evaluation for fixed ordered chemical features.

The official encoders do not consume ligand coordinates. Cache their Gaussian
parameters, not sampled scores, and retain official mixed-precision arithmetic.
Torch is imported lazily so ordinary CPU preparation needs no Torch install.
"""
import hashlib
import numpy as np


def canonical_coordinate_molecule(template, coordinates):
    """Reproduce the official SDF read, including coordinate-derived stereo."""
    from rdkit import Chem
    mol = Chem.Mol(template)
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid chemical coordinate array')
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(xyz):
        conformer.SetAtomPosition(i, pos)
    mol.AddConformer(conformer)
    result = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=True)
    if result is None:
        raise ValueError('Coordinate molecule failed official SDF preparation')
    return result


def chemistry_signature(mol):
    """Ordered chemistry/stereo fields underlying the official graph features."""
    from rdkit import Chem
    centers = dict(Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                           useLegacyImplementation=False))
    atoms = [(a.GetSymbol(), a.GetDegree(), a.GetFormalCharge(), a.GetNumRadicalElectrons(),
              str(a.GetHybridization()), a.GetIsAromatic(), a.GetTotalNumHs(), centers.get(a.GetIdx()))
             for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()),
              b.GetIsConjugated(), b.IsInRing(), str(b.GetStereo())) for b in mol.GetBonds()]
    return hashlib.sha256(repr((atoms, bonds)).encode()).hexdigest()


def feature_signature(graph, node_key, edge_key):
    """Hash the ordered tensors consumed by the official graph encoder."""
    h = hashlib.sha256()
    tensors = [graph.ndata[node_key].float(), graph.edata[edge_key].float(), *graph.edges()]
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str((arr.shape, arr.dtype.str)).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


class CachedRTM:
    def __init__(self, pi, sigma, mu, positions, signature=None):
        import torch
        if pi.ndim != 3 or pi.shape != sigma.shape or pi.shape != mu.shape:
            raise ValueError('Expected atom/residue/Gaussian parameter arrays')
        if positions.shape != (pi.shape[1], 24, 3):
            raise ValueError('Expected official 24-atom padded residue coordinates')
        if not all(torch.isfinite(x).all() for x in (pi, sigma, mu)):
            raise ValueError('Nonfinite mixture parameters')
        if (pi <= 0).any() or (sigma <= 0).any():
            raise ValueError('Invalid mixture probabilities or standard deviations')
        if not torch.isfinite(positions).all(-1).any(-1).all():
            raise ValueError('A residue has no finite atom coordinates')
        self.pi, self.sigma, self.mu = pi.detach(), sigma.detach(), mu.detach()
        self.positions = positions.detach().double()
        self.signature = signature
        self.device = pi.device
        if any(t.device != self.device for t in (sigma, mu, positions)):
            raise ValueError('All cached tensors must share a device')
        self.flat_positions = self.positions.reshape(-1, 3)
        self.y_squared = (self.flat_positions ** 2).sum(-1)

    @classmethod
    def from_model(cls, model, protein, ligand, device='cuda'):
        return BatchShapeRTM(model, protein, ligand, device)

    def to(self, device):
        return type(self)(self.pi.to(device), self.sigma.to(device), self.mu.to(device),
                          self.positions.to(device), self.signature)

    def score(self, coordinates, chunk_size=64):
        import torch
        from torch.distributions import Normal
        xyz = np.asarray(coordinates, dtype=np.float64)
        if xyz.ndim != 3 or xyz.shape[1:] != (self.pi.shape[0], 3) or not np.isfinite(xyz).all():
            raise ValueError('Invalid ordered ligand coordinate batch')
        if chunk_size < 1:
            raise ValueError('Invalid chunk size')
        result = []
        with torch.no_grad():
            for offset in range(0, len(xyz), chunk_size):
                # copy handles negative strides and ensures caller cannot change an async input.
                x = torch.as_tensor(xyz[offset:offset + chunk_size].copy(), device=self.device)
                count = x.shape[0]
                y = self.flat_positions.T.unsqueeze(0).expand(count, -1, -1)
                d2 = -2 * torch.bmm(x, y) + self.y_squared[None, None, :]
                d2 = d2 + (x ** 2).sum(-1).unsqueeze(-1)
                # Match official NaN treatment, including tiny negative d2; no clamp.
                distance = torch.nan_to_num(d2 ** .5, nan=10000).reshape(
                    count, self.pi.shape[0], self.pi.shape[1], 24).min(-1)[0]
                logprob = Normal(self.mu, self.sigma).log_prob(distance.unsqueeze(-1))
                logprob += torch.log(self.pi)
                density = logprob.exp().sum(-1)
                density[distance > 5.] = 0.
                values = density.sum((1, 2)).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite RTMScore prediction')
                result.append(values)
        return np.concatenate(result) if result else np.empty(0, dtype=float)


class BatchShapeRTM:
    """Preserve the official FP32 encoder's inference-batch shape.

    Encoder inputs are coordinate independent, but floating-point GEMM kernels
    can depend on batch shape. Cache a separate parameter bank per actual batch
    size, using the official DataLoader's 64-pose chunks. Do not relax numerical
    tolerances or assume a singleton bank equals a batched encoder invocation.
    """
    def __init__(self, model, protein, ligand, device):
        import torch
        if model.training:
            raise ValueError('Cached inference requires model.eval()')
        self.signature = dict(ligand=feature_signature(ligand, 'atom', 'bond'),
                              protein=feature_signature(protein, 'feats', 'feats'))
        self.model, self.protein, self.ligand = model, protein, ligand
        self.device = torch.device(device)
        self.atoms, self.residues = ligand.num_nodes(), protein.num_nodes()
        self.banks = {}

    def _bank(self, count):
        import torch
        import dgl
        if count in self.banks:
            return self.banks[count]
        if self.model is None:
            raise ValueError('Only available coefficient banks can be used by a device clone')
        if self.model.training:
            raise ValueError('Cached inference requires model.eval()')
        protein = dgl.batch([self.protein] * count).to(self.device)
        ligand = dgl.batch([self.ligand] * count).to(self.device)
        with torch.no_grad():
            pi, sigma, mu, *_ = self.model(protein, ligand)
        parameters = []
        for parameter in (pi, sigma, mu):
            shaped = parameter.reshape(count, self.atoms, self.residues, -1)
            if not torch.equal(shaped, shaped[0:1].expand_as(shaped)):
                raise RuntimeError('Identical chemical features produced unequal batch slots')
            parameters.append(shaped[0].clone())
        positions = self.protein.ndata['pos'].to(self.device)
        bank = CachedRTM(*parameters, positions, self.signature)
        self.banks[count] = bank
        return bank

    def score(self, coordinates, chunk_size=64):
        xyz = np.asarray(coordinates, dtype=np.float64)
        if xyz.ndim != 3 or xyz.shape[1:] != (self.atoms, 3) or not np.isfinite(xyz).all():
            raise ValueError('Invalid ordered ligand coordinate batch')
        if chunk_size != 64:
            raise ValueError('Official reference coefficient banks require chunk_size=64')
        values = []
        for offset in range(0, len(xyz), chunk_size):
            part = xyz[offset:offset+chunk_size]
            values.append(self._bank(len(part)).score(part))
        return np.concatenate(values) if values else np.empty(0, dtype=float)

    def to(self, device):
        import torch
        clone = object.__new__(type(self))
        clone.model = None  # CPU benchmark must use the very same GPU coefficients.
        clone.protein, clone.ligand = self.protein, self.ligand
        clone.atoms, clone.residues, clone.signature = self.atoms, self.residues, self.signature
        clone.device = torch.device(device)
        clone.banks = {count: bank.to(device) for count, bank in self.banks.items()}
        return clone
