#!/usr/bin/env python3
"""Dock a new ligand into a protein using CPU Vina and GPU RTMScore hybrid search."""
import argparse
import math
import os
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Box lengths and coordinates are in Angstrom. Reference ligand must already be in the receptor frame. '
        'Use a new output directory. See README.md for examples.')
    parser.add_argument('--receptor', type=Path, required=True, help='Protein PDB file')
    parser.add_argument('--ligand', type=Path, required=True, help='One molecule in SDF or MOL2')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    region = parser.add_mutually_exclusive_group(required=True)
    region.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='Manual box center')
    region.add_argument('--reference-ligand', type=Path, help='Bound ligand SDF/MOL2; used only to define region')
    parser.add_argument('--size', type=float, nargs=3, metavar=('SX', 'SY', 'SZ'), help='Manual box full side lengths')
    parser.add_argument('--padding', type=float, default=6., help='Reference-ligand box padding per side, default 6 A')
    parser.add_argument('--receptor-chains', nargs='+', help='Receptor chain IDs, default all supplied chains')
    parser.add_argument('--remove-residues', nargs='+', default=[], help='Explicit residue names of free nonprotein ligands to remove; waters removed automatically')
    parser.add_argument('--model', type=int, help='One-based receptor model index, required for multi-model PDB')
    parser.add_argument('--seed', type=int, default=20261021)
    for name, default in dict(chains=32, warmup=4, steps=256, restarts=4,
                              local_steps=300, relax_workers=8, num_poses=20).items():
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--weight', type=float, default=1.)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--translation', type=float, default=2.)
    parser.add_argument('--prepare-only', action='store_true', help='Prepare and validate files without running search')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    parser.add_argument('--rtm-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--rtm-python', type=Path,
        default=Path(os.environ.get('RTMSCORE_PYTHON', sys.executable)),
        help='RTMScore worker interpreter')
    parser.add_argument('--obabel', type=Path,
        default=Path(os.environ.get('VINA_RTMSCORE_OBABEL', shutil.which('obabel') or 'obabel')),
        help='Open Babel executable')
    args = parser.parse_args(argv)
    if args.center is not None:
        if args.size is None:
            parser.error('--center requires --size')
        if not all(math.isfinite(v) for v in args.center) or not all(math.isfinite(v) and v > 0 for v in args.size):
            parser.error('Box center must be finite and side lengths positive and finite')
    elif args.size is not None:
        parser.error('--size is only valid with --center')
    if not 1 <= args.seed <= 2147483647:
        parser.error('--seed must be in [1, 2147483647]')
    if args.model is not None and args.model < 1:
        parser.error('--model must be positive')
    for name in ('chains', 'warmup', 'steps', 'restarts', 'local_steps', 'relax_workers', 'num_poses'):
        if getattr(args, name) < 1:
            parser.error('--'+name.replace('_', '-')+' must be positive')
    for name in ('temperature', 'translation', 'padding'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error('--'+name+' must be finite and positive')
    if not math.isfinite(args.weight) or args.weight < 0:
        parser.error('--weight must be finite and nonnegative')
    if args.rtm_root is None:
        args.rtm_root = args.root/'third_party/RTMScore'
    return args


def main(argv=None):
    args = parse_args(argv)
    from vina_rtmscore.docking import run
    try:
        record = run(args)
    except (ValueError, OSError) as exc:
        print(f'Docking error: {exc}', file=sys.stderr)
        return 1
    if record['status'] == 'failed':
        print('Docking failed: '+record.get('error', record.get('cleanup_error', 'see run.json')), file=sys.stderr)
        print('Details: '+str(args.out.resolve()/'run.json'), file=sys.stderr)
        return 1
    print(f"Status: {record['status']}\nOutput: {args.out.resolve()}")
    if record['status'] == 'completed':
        print('Top pose: '+str(args.out.resolve()/'selected.sdf'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
