"""Standalone docking of a new ligand into a user-supplied protein."""
import csv
import gzip
import json
import os
from pathlib import Path
import time
import traceback

from rdkit import Chem

from vina_rtmscore.preparation import digest, prepare_inputs
from vina_rtmscore.objective import Calibration, objective
from vina_rtmscore.chemistry import molecule_from_pose


CONFIG_FIELDS = ('chains', 'warmup', 'steps', 'restarts', 'local_steps', 'relax_workers',
                 'weight', 'temperature', 'translation', 'seed')


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def claim_output(directory):
    Path(directory).mkdir(parents=True, exist_ok=False)


def write_results(directory, archive, template, mapping, calibration, weight, num_poses):
    directory = Path(directory)
    ranked = sorted(archive, key=lambda s: (objective(s['energy'], s['rtm'], 'hybrid', calibration, weight), s['id']))
    if not ranked or len({s['id'] for s in ranked}) != len(ranked):
        raise ValueError('Missing or duplicate candidate identities')
    columns = ['rank', 'pose_id', 'vina_energy', 'rtmscore', 'hybrid_objective', 'origin', 'chain', 'restart', 'step']
    with (directory/'scores.csv').open('w', newline='') as table, \
            gzip.open(directory/'evaluated.sdf.gz', 'wt', compresslevel=3) as complete, \
            Chem.SDWriter(str(directory/'poses.sdf')) as top, \
            Chem.SDWriter(str(directory/'selected.sdf')) as selected:
        csvwriter = csv.DictWriter(table, fieldnames=columns); csvwriter.writeheader()
        allwriter = Chem.SDWriter(complete)
        try:
            for rank, state in enumerate(ranked, 1):
                row = dict(rank=rank, pose_id=state['id'], vina_energy=state['energy'], rtmscore=state['rtm'],
                    hybrid_objective=objective(state['energy'], state['rtm'], 'hybrid', calibration, weight),
                    **{k: state[k] for k in ('origin', 'chain', 'restart', 'step')})
                csvwriter.writerow(row)
                mol = molecule_from_pose(template, state['pose'], mapping, state['id'])
                for key, value in row.items():
                    if isinstance(value, float):
                        mol.SetDoubleProp(key, value)
                    elif isinstance(value, int):
                        mol.SetIntProp(key, value)
                    else:
                        mol.SetProp(key, value)
                allwriter.write(mol)
                if rank <= num_poses:
                    top.write(mol)
                if rank == 1:
                    selected.write(mol)
        finally:
            allwriter.close()
    return dict(num_candidates=len(ranked), num_output_poses=min(num_poses, len(ranked)),
                selected_pose_id=ranked[0]['id'])


def source_fingerprints(root):
    root = Path(root)
    files = sorted((root/'vina_rtmscore').glob('*.py'))
    files += [root/name for name in (
        'scripts/dock.py', 'scripts/rtm_worker.py', 'scripts/build_fast_vina.py',
        'native/fast_vina.cpp', 'native/build_provenance.json',
        'native/vina-1.2.7.tar.gz', 'native/libfast_vina.so')]
    return {str(path.relative_to(root)): digest(path) for path in files}


def verify_prepared_inputs(directory, audit):
    for item in audit['inputs'].values():
        if digest(item['copied']) != item['sha256']:
            raise ValueError('Copied input changed during computation: '+item['copied'])
    for name, expected in audit['prepared_hashes'].items():
        if digest(Path(directory)/'prepared'/name) != expected:
            raise ValueError('Prepared input changed during computation: '+name)


def run(args):
    from vina_rtmscore.fast_vina import FastVinaBatchRelaxer
    from vina_rtmscore.backends import ChemicallyCheckedRelaxer, CoordinateRTMClient
    from vina_rtmscore.engine import search
    from vina_rtmscore.provenance import rtm_source_hashes
    root = args.root.resolve(); directory = args.out.resolve()
    for path in [args.receptor, args.ligand] + ([args.reference_ligand] if args.reference_ligand else []):
        if not path.is_file():
            raise ValueError('Input file does not exist: '+str(path))
    if args.receptor.suffix.lower() != '.pdb':
        raise ValueError('Receptor must be a .pdb file')
    if not args.obabel.is_file() or not os.access(args.obabel, os.X_OK):
        raise ValueError('Open Babel executable unavailable; specify --obabel')
    if not args.rtm_python.is_file():
        raise ValueError('RTMScore Python unavailable; specify --rtm-python')
    claim_output(directory)
    config = {key: getattr(args, key) for key in CONFIG_FIELDS}
    record = dict(schema_version=1, status='running', mode='custom_file_docking', config=config,
        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), output_directory=str(directory),
        region_mode='reference_ligand' if args.reference_ligand else 'manual',
        prepare_only=args.prepare_only, num_requested_poses=args.num_poses,
        score_directions=dict(vina_energy='lower', rtmscore='higher', hybrid_objective='lower'),
        local_optimizer='official Vina1.2.7 CPU', neural_scorer='RTMScore1 cached CUDA',
        score_interpretation='Search scores, not measured affinity; no native reference or RMSD required')
    atomic_json(directory/'run.json', record)
    begin = time.monotonic(); client = relax = None
    try:
        record['source_hashes'] = source_fingerprints(root)
        record['rtm_source_hashes'] = rtm_source_hashes(args.rtm_root)
        record['checkpoint_sha256'] = digest(args.rtm_root/'trained_models/rtmscore_model1.pth')
        record['native_library'] = dict(path=str(Path(os.environ.get('FAST_VINA_LIBRARY', str(root/'native/libfast_vina.so'))).resolve()))
        record['native_library']['sha256'] = digest(record['native_library']['path'])
        template, start, mapping, audit = prepare_inputs(args, directory)
        record['box'] = audit['box']; record['preparation_sha256'] = digest(directory/'preparation.json')
        record['preparation_elapsed_s'] = time.monotonic()-begin
        atomic_json(directory/'run.json', record)
        if args.prepare_only:
            record['status'] = 'prepared'
        else:
            print('Preparation complete. Starting CPU Vina / GPU RTMScore hybrid search.', flush=True)
            native = FastVinaBatchRelaxer(directory/'prepared/receptor.pdbqt', start,
                audit['box']['center'], audit['box']['size'], directory, max_steps=args.local_steps,
                cpu=1, seed=args.seed, workers=args.relax_workers)
            relax = ChemicallyCheckedRelaxer(native, template, mapping)
            client = CoordinateRTMClient(str(args.rtm_python), args.rtm_root, directory,
                                        directory/'prepared/pocket.pdb', template, mapping)
            with gzip.open(directory/'trace.jsonl.gz', 'wt', compresslevel=3) as trace, \
                    gzip.open(directory/'evaluations.jsonl.gz', 'wt', compresslevel=3) as journal:
                def emit(row):
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    if row.get('chain') == config['chains']-1:
                        trace.flush(); journal.flush()
                        atomic_json(directory/'progress.json', dict(phase=row['phase'], restart=row.get('restart'),
                            step=row.get('step'), elapsed_s=time.monotonic()-begin))
                def checkpoint(row):
                    journal.write(json.dumps(row, allow_nan=False)+'\n')
                archive, meta = search(start, audit['box']['center'], audit['box']['size'],
                                       relax, client, config, emit, checkpoint)
            if not client.verification or not client.verification[0]['passed']:
                raise ValueError('Official RTMScore equivalence verification missing')
            calibration = Calibration(**meta['calibration'])
            record.update(write_results(directory, archive, template, mapping, calibration, args.weight, args.num_poses))
            record.update(search=meta, backend=relax.stats, gpu_timings=client.timings,
                          scorer_verification=client.verification, status='completed')
        verify_prepared_inputs(directory, audit)
        if source_fingerprints(root) != record['source_hashes'] or rtm_source_hashes(args.rtm_root) != record['rtm_source_hashes'] or digest(args.rtm_root/'trained_models/rtmscore_model1.pth') != record['checkpoint_sha256'] or digest(record['native_library']['path']) != record['native_library']['sha256']:
            raise ValueError('Source, model or native library changed during computation')
        if digest(directory/'preparation.json') != record['preparation_sha256']:
            raise ValueError('Preparation manifest changed during computation')
        outputs = ['preparation.json'] if args.prepare_only else [
            'preparation.json', 'selected.sdf', 'poses.sdf', 'scores.csv', 'evaluated.sdf.gz',
            'trace.jsonl.gz', 'evaluations.jsonl.gz']
        record['output_hashes'] = {name: digest(directory/name) for name in outputs}
    except BaseException as exc:
        (directory/'error.log').write_text(traceback.format_exc())
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        for backend in (client, relax):
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    record.update(status='failed', cleanup_error=f'{type(exc).__name__}: {exc}')
        record['elapsed_s'] = time.monotonic()-begin
        record['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        atomic_json(directory/'run.json', record)
    return record
