"""Protein-only preparation helpers; no structure downloads or datasets."""
from collections import defaultdict
import json
import subprocess

SIDECHAIN_ATOMS = {
    'GLY':'', 'ALA':'CB', 'VAL':'CB CG1 CG2', 'LEU':'CB CG CD1 CD2',
    'ILE':'CB CG1 CG2 CD1', 'SER':'CB OG', 'THR':'CB OG1 CG2',
    'CYS':'CB SG', 'MET':'CB CG SD CE', 'PRO':'CB CG CD',
    'ASP':'CB CG OD1 OD2', 'ASN':'CB CG OD1 ND2',
    'GLU':'CB CG CD OE1 OE2', 'GLN':'CB CG CD OE1 NE2',
    'LYS':'CB CG CD CE NZ', 'ARG':'CB CG CD NE CZ NH1 NH2',
    'HIS':'CB CG ND1 CD2 CE1 NE2', 'PHE':'CB CG CD1 CD2 CE1 CE2 CZ',
    'TYR':'CB CG CD1 CD2 CE1 CE2 CZ OH',
    'TRP':'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2'}


def selected_atoms(residue):
    # Shared blank-altloc atoms plus the highest-occupancy residue conformer.
    labels = defaultdict(float)
    for atom in residue:
        if atom.altloc not in ('\x00', ' ', '.'):
            labels[atom.altloc] += atom.occ
    alt = max(labels, key=lambda k: (labels[k], -ord(k))) if labels else None
    result = {}
    for atom in residue:
        if atom.altloc in ('\x00', ' ', '.') or atom.altloc == alt:
            if atom.name not in result or atom.occ > result[atom.name].occ:
                result[atom.name] = atom
    return list(result.values()), alt


def prepare_receptor(pdb_path, pdbqt_path, obabel, log_path):
    # Open Babel's -xr disables the torsion tree (rigid receptor); -p7.4 adds
    # pH-adjusted H and Gasteiger provides explicit partial charges.
    command = [obabel, '-ipdb',str(pdb_path),'-opdbqt','-O',str(pdbqt_path),
               '-p','7.4','--partialcharge','gasteiger','-xr']
    result = subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log_path.write_text(json.dumps(command)+'\n'+result.stdout)
    if result.returncode or not pdbqt_path.exists():
        raise ValueError('Receptor Open Babel preparation failed; see '+str(log_path))
    text = pdbqt_path.read_text()
    if not any(l.startswith(('ATOM  ','HETATM')) for l in text.splitlines()) or 'BRANCH' in text or 'ROOT' in text:
        raise ValueError('Invalid rigid receptor PDBQT')
