#!/usr/bin/env python
"""Persistent exact RTMScore CUDA coefficient cache and coordinate-only IPC."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rtm-root', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    root = args.rtm_root.resolve()
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'example'))
    import RTMScore as bundled_rtm
    if {Path(path).resolve() for path in bundled_rtm.__path__} != {(root/'RTMScore').resolve()}:
        raise RuntimeError('RTMScore resolved outside the requested inference source root')
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from rdkit import Chem
        from torch.utils.data import DataLoader
        from RTMScore.feats.mol2graph_rdmda_res import load_mol, prot_to_graph, mol_to_graph
        from RTMScore.data.data import PDBbindDataset
        from RTMScore.model.utils import collate, run_an_eval_epoch
        from vina_rtmscore.cached_rtm import (CachedRTM, feature_signature,
                                           canonical_coordinate_molecule, chemistry_signature)
        from vina_rtmscore.model_loader import load_model
        torch.set_num_threads(2); torch.manual_seed(20261021)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        # Match the precision of the existing official scorer; never use AMP.
        model, checkpoint = load_model(root, args.device)
    cache = template = protein = signature = chemical_key = None

    def official(coords):
        mols = []
        for i, xyz in enumerate(coords):
            mol = canonical_coordinate_molecule(template, xyz)
            mol.SetProp('_Name', f'verify{i}')
            mols.append(mol)
        graphs = [mol_to_graph(m, explicit_H=False, use_chirality=True) for m in mols]
        if any(feature_signature(g, 'atom', 'bond') != signature for g in graphs):
            raise ValueError('Chemical features changed across coordinate-only requests')
        data = PDBbindDataset(ids=[str(i) for i in range(len(mols))], ligs=graphs,
                              prots=[protein] * len(mols))
        loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
        reference = np.asarray(run_an_eval_epoch(model, loader, pred=True, dist_threhold=5,
                                                device=args.device)).reshape(-1)
        return mols, reference

    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['op'] == 'initialize':
                    template = Chem.SDMolSupplier(request['template_sdf'], removeHs=True)[0]
                    if template is None:
                        raise ValueError('Invalid chemical template')
                    chemical_key = chemistry_signature(template)
                    protein = prot_to_graph(load_mol(request['pocket'], explicit_H=False, use_chirality=True), 10.)
                    ligand = mol_to_graph(template, explicit_H=False, use_chirality=True)
                    cache = CachedRTM.from_model(model, protein, ligand, args.device)
                    signature = cache.signature['ligand']
                    response = dict(initialized=True, signature=cache.signature,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        device=args.device, torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None)
                elif request['op'] == 'score':
                    if cache is None:
                        raise ValueError('Initialize the fixed chemical template first')
                    coords = np.asarray(request['coordinates'], dtype=np.float64)
                    ids = request['ids']
                    if len(ids) != len(coords) or len(set(ids)) != len(ids) or not len(ids):
                        raise ValueError('Invalid pose identities')
                    # Every batch reproduces official coordinate-derived stereo.
                    # Only encoder work is cached; incompatible chemistry is never reused.
                    for xyz in coords:
                        mol = canonical_coordinate_molecule(template, xyz)
                        if chemistry_signature(mol) != chemical_key:
                            raise ValueError('Chemical feature signature changed in a coordinate request')
                    start = time.perf_counter()
                    values = cache.score(coords)
                    response = dict(ids=ids, scores=values.tolist(), elapsed_s=time.perf_counter()-start)
                    if request.get('verify') or request.get('profile'):
                        start = time.perf_counter(); mols, reference = official(coords)
                        reference_s = time.perf_counter() - start
                        delta = float(np.max(np.abs(values-reference)))
                        if not np.allclose(values, reference, atol=1e-6, rtol=1e-6):
                            raise ValueError(f'Cached score differs from official model: {delta}')
                        # Check the public API too, including chemical feature preparation.
                        from rtmscore import scoring
                        with tempfile.TemporaryDirectory(prefix='rtm_v2_verify_') as directory:
                            sdf = Path(directory)/'poses.sdf'
                            with Chem.SDWriter(str(sdf)) as writer:
                                for mol in mols: writer.write(mol)
                            actual_ids, public = scoring(prot=request['pocket'], lig=str(sdf),
                                modpath=str(checkpoint), cut=10., gen_pocket=False,
                                explicit_H=False, use_chirality=True, parallel=False, batch_size=64,
                                dist_threhold=5, device=args.device, num_workers=0,
                                num_node_featsp=41, num_node_featsl=41, num_edge_featsp=5,
                                num_edge_featsl=10, hidden_dim0=128, hidden_dim=128,
                                n_gaussians=10, dropout_rate=.1)
                        if list(actual_ids) != [f'verify{i}-{i}' for i in range(len(coords))]:
                            raise ValueError('Public API pose identity mismatch')
                        public = np.asarray(public).reshape(-1)
                        public_delta = float(np.max(np.abs(values-public)))
                        if not np.allclose(values, public, atol=1e-6, rtol=1e-6):
                            raise ValueError(f'Cached score differs from public API: {public_delta}')
                        response['verification'] = dict(passed=True, max_abs_difference=delta,
                            public_api_max_abs_difference=public_delta, poses=len(coords),
                            official_graph_scoring_s=reference_s)
                    if request.get('profile'):
                        cpu_cache = cache.to('cpu')
                        cpu_values = cpu_cache.score(coords)
                        cpu_delta = float(np.max(np.abs(values-cpu_values)))
                        timings = {}
                        for name, evaluator in [('cached_cpu', cpu_cache), ('cached_gpu', cache)]:
                            evaluator.score(coords)
                            samples = []
                            for _ in range(10):
                                start = time.perf_counter(); evaluator.score(coords)
                                samples.append(time.perf_counter()-start)
                            timings[name] = dict(median_s=float(np.median(samples)), samples_s=samples,
                                                 poses_per_s=len(coords)/float(np.median(samples)))
                        response['profile'] = dict(timings=timings, batch_size=len(coords),
                            cpu_gpu_max_abs_difference=cpu_delta, device=args.device,
                            memory_peak_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
                else:
                    raise ValueError('Unknown worker operation')
            print(json.dumps(response, allow_nan=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')), flush=True)


if __name__ == '__main__':
    main()
