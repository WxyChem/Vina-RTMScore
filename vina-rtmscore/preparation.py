"""Prepare user files for hybrid docking without a benchmark/native manifest."""
import hashlib
import json
from pathlib import Path
import shutil

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from vina_rtmscore.receptor import SIDECHAIN_ATOMS, selected_atoms, prepare_receptor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_box(center=None, size=None, reference_points=None, padding=6.):
    if reference_points is not None:
        if center is not None or size is not None:
            raise ValueError('Manual box and reference ligand are mutually exclusive')
        xyz = np.asarray(reference_points, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError('Invalid reference ligand coordinates')
        if not np.isfinite(padding) or padding <= 0:
            raise ValueError('Reference padding must be finite and positive')
        lo, hi = xyz.min(0), xyz.max(0)
        center, size, mode = (lo+hi)/2, hi-lo+2*padding, 'reference_ligand'
    else:
        if center is None or size is None:
            raise ValueError('Supply both --center and --size, or --reference-ligand')
        mode = 'manual'
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError('Box requires three finite center coordinates and positive finite side lengths')
    return dict(mode=mode, center=center.tolist(), size=size.tolist())


def read_ligand(path, check_stereo=True):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Input ligand file does not exist: '+str(path))
    suffix = path.suffix.lower()
    if suffix == '.sdf':
        molecules = list(Chem.SDMolSupplier(str(path), removeHs=False, strictParsing=True))
        if len(molecules) != 1:
            raise ValueError('Ligand file must contain exactly one molecule')
        mol = molecules[0]
    elif suffix == '.mol2':
        if path.read_text().upper().count('@<TRIPOS>MOLECULE') != 1:
            raise ValueError('Ligand file must contain exactly one MOL2 molecule')
        mol = Chem.MolFromMol2File(str(path), removeHs=False, cleanupSubstructures=False)
    else:
        raise ValueError('Ligand must be a single-molecule .sdf or .mol2 file')
    if mol is None or not mol.GetNumHeavyAtoms():
        raise ValueError('Cannot parse ligand chemistry: '+str(path))
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Disconnected ligand fragments are unsupported; supply one explicit molecular species')
    mol = Chem.RemoveHs(mol)
    # The fixed-feature RTM client and Meeko map require a heavy-only template.
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError('Unsupported retained explicit/isotopic/stereo-defining hydrogen; no atoms were silently removed')
    if any(a.GetAtomicNum() not in {6, 7, 8, 9, 15, 16, 17, 35, 53} or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('Unsupported ligand element or radical for this Vina/RTMScore interface')
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if check_stereo:
        if any(group.GetGroupType() != Chem.StereoGroupType.STEREO_ABSOLUTE for group in mol.GetStereoGroups()):
            raise ValueError('Relative/racemic enhanced stereochemistry is unsupported; supply a specified stereoisomer')
        unresolved = [str(s.centeredOn) for s in Chem.FindPotentialStereo(mol)
                      if s.specified != Chem.StereoSpecified.Specified or
                      s.type not in {Chem.StereoType.Atom_Tetrahedral, Chem.StereoType.Bond_Double}]
        if unresolved:
            raise ValueError('Unspecified ligand stereochemistry at atom/bond indices: '+', '.join(unresolved))
        # SDF writers may reverse an undirected bond to put its stereocenter first.
        # The existing fixed-feature scorer hashes ordered bond endpoints, so bind
        # the template to that serialized representation before preparing Meeko.
        normalized = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=True)
        def identity(molecule):
            if molecule is None:
                return None
            atoms = [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), a.GetTotalNumHs(),
                      a.GetNumRadicalElectrons(), a.GetIsAromatic(),
                      a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else None) for a in molecule.GetAtoms()]
            bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                            str(b.GetBondType()), b.GetIsAromatic(), str(b.GetStereo())) for b in molecule.GetBonds())
            return atoms, bonds, Chem.MolToSmiles(molecule)
        if identity(normalized) != identity(mol):
            raise ValueError('SDF template normalization changed ligand chemistry/stereochemistry')
        mol = normalized
    return mol


def reference_coordinates(path):
    mol = read_ligand(path, check_stereo=False)
    if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError('Reference ligand requires 3D coordinates in the receptor frame')
    xyz = np.asarray(mol.GetConformer().GetPositions())
    if not np.isfinite(xyz).all() or (len(xyz) > 1 and np.max(np.ptp(xyz, axis=0)) < 1e-4):
        raise ValueError('Reference ligand requires finite noncollapsed 3D coordinates')
    return xyz


def prepare_start(template, path, seed):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina_rtmscore.pose_moves import Pose
    from vina_rtmscore.chemistry import molecule_from_pose
    from vina_rtmscore.cached_rtm import chemistry_signature, canonical_coordinate_molecule
    mol = Chem.AddHs(Chem.Mol(template)); mol.RemoveAllConformers()
    parameters = AllChem.ETKDGv3(); parameters.randomSeed = seed
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise ValueError('Independent ETKDG embedding failed')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    else:
        raise ValueError('Ligand has no complete MMFF/UFF parameterization')
    setups = MoleculePreparation().prepare(mol)
    if len(setups) != 1:
        raise ValueError('Expected exactly one Meeko ligand setup')
    text, ok, error = PDBQTWriterLegacy.write_string(setups[0], add_index_map=True)
    if not ok:
        raise ValueError('Meeko ligand preparation: '+error)
    mapping = {}
    for line in text.splitlines():
        if line.startswith('REMARK INDEX MAP'):
            values = [int(v) for v in line.split()[3:]]
            if len(values) % 2:
                raise ValueError('Malformed Meeko atom map')
            mapping.update(zip(values[::2], values[1::2]))
    count = template.GetNumAtoms()
    mapping = {i: mapping[i] for i in range(1, count+1) if i in mapping}
    pose = Pose.parse(text)
    if len(mapping) != count or len(set(mapping.values())) != count or not set(mapping.values()) <= set(pose.serials):
        raise ValueError('Incomplete one-to-one Meeko heavy-atom map')
    restored = molecule_from_pose(template, pose, mapping, 'independent_start')
    actual = canonical_coordinate_molecule(template, restored.GetConformer().GetPositions())
    if chemistry_signature(actual) != chemistry_signature(template):
        raise ValueError('Independent ligand geometry changed specified chemistry/stereochemistry')
    Path(path).write_text(text)
    return pose, mapping


def residue_id(chain, residue):
    return dict(chain=chain.name, seqid=str(residue.seqid), residue=residue.name)


def select_receptor(path, chains=None, model=None, remove_residues=None):
    source = gemmi.read_structure(str(path))
    if not len(source):
        raise ValueError('Empty receptor structure')
    if model is None and len(source) != 1:
        raise ValueError('Multiple receptor models; specify --model (one-based index)')
    model = 1 if model is None else model
    if not 1 <= model <= len(source):
        raise ValueError('Receptor model index out of range')
    source.merge_chain_parts()
    original = source[model-1]
    available = [c.name for c in original]
    names = available if chains is None else chains
    if not names or len(names) != len(set(names)) or not set(names) <= set(available):
        raise ValueError('Invalid or duplicate receptor chain selection')
    remove_residues = set(remove_residues or [])
    retained_connections = []
    for connection in source.connections:
        if connection.type not in {gemmi.ConnectionType.Covale, gemmi.ConnectionType.Disulf}:
            continue
        partners = [connection.partner1, connection.partner2]
        kept = [p.chain_name in names and p.res_id.name in SIDECHAIN_ATOMS for p in partners]
        if any(kept) and not all(kept):
            raise ValueError('Unsupported covalent receptor connection: '+str(connection.partner1)+' -- '+str(connection.partner2))
        if all(kept):
            retained_connections.append(connection)
    atoms_by_serial = {a.serial: (c.name in names and r.name in SIDECHAIN_ATOMS)
                       for c in original for r in c for a in r}
    for serial, neighbors in source.conect_map.items():
        for neighbor in neighbors:
            if serial in atoms_by_serial and neighbor in atoms_by_serial and atoms_by_serial[serial] != atoms_by_serial[neighbor]:
                raise ValueError('Unsupported covalent receptor CONECT between retained and removed atoms')
    output = gemmi.Structure(); output.name = 'custom_receptor'
    selected_model = gemmi.Model(1)
    audit = dict(model=model, selected_chains=names, removed_residues=[], alternate_conformations=[],
                 excluded_chains=[c for c in available if c not in names], removed_hydrogens=0,
                 retained_connections=[dict(type=str(c.type), partner1=str(c.partner1), partner2=str(c.partner2)) for c in retained_connections])
    cofactors = {'HEM', 'HEC', 'HEA', 'NAD', 'NAP', 'NDP', 'FAD', 'FMN', 'SAM', 'SAH', 'PLP', 'COA', 'ATP', 'ADP', 'GTP', 'GDP', 'TPP', 'BTN', 'F42'}
    for original_chain in original:
        if original_chain.name not in names:
            continue
        chain = gemmi.Chain(original_chain.name)
        seen = set()
        for residue in original_chain:
            identity = residue_id(original_chain, residue)
            if residue.name not in SIDECHAIN_ATOMS:
                if gemmi.find_tabulated_residue(residue.name).is_amino_acid() or residue.name in cofactors or any(a.element.is_metal for a in residue):
                    raise ValueError('Unsupported modified residue/metal/cofactor: '+json.dumps(identity))
                if not gemmi.find_tabulated_residue(residue.name).is_water() and residue.name not in remove_residues:
                    raise ValueError('Unclassified nonprotein residue: '+json.dumps(identity)+
                                     '; specify --remove-residues '+residue.name+' only if this is a free ligand to remove')
                audit['removed_residues'].append(identity)
                continue
            if str(residue.seqid) in seen:
                raise ValueError('Duplicate receptor residue identity: '+json.dumps(identity))
            seen.add(str(residue.seqid))
            atoms, alt = selected_atoms(residue)
            if alt is not None:
                audit['alternate_conformations'].append(dict(**identity, selected_altloc=alt))
            selected = gemmi.Residue(); selected.name = residue.name
            selected.seqid = residue.seqid; selected.het_flag = 'A'
            for atom in atoms:
                if atom.element.is_hydrogen:
                    audit['removed_hydrogens'] += 1
                    continue
                if not np.isfinite(list(atom.pos)).all():
                    raise ValueError('Nonfinite receptor coordinates: '+json.dumps(identity))
                copy = atom.clone(); copy.altloc = '\x00'; selected.add_atom(copy)
            expected = set(('N CA C O '+SIDECHAIN_ATOMS[residue.name]).split())
            if not expected <= {a.name for a in selected}:
                raise ValueError('Incomplete receptor residue; supply a repaired PDB: '+json.dumps(identity))
            if any(a.name not in expected | {'OXT'} for a in selected):
                raise ValueError('Unsupported extra heavy atoms in receptor residue: '+json.dumps(identity))
            chain.add_residue(selected)
        if len(chain):
            selected_model.add_chain(chain)
    if not len(selected_model):
        raise ValueError('No standard protein residues in selected receptor chains')
    output.add_model(selected_model)
    for connection in retained_connections:
        output.connections.append(connection)
    return output, audit


def write_custom_pocket(structure, path, box, reference_points=None, radius=10.):
    pocket = structure.clone(); retained = []
    center, half = np.asarray(box['center']), np.asarray(box['size'])/2
    for chain in pocket[0]:
        remove = []
        for index, residue in enumerate(chain):
            xyz = np.asarray([list(a.pos) for a in residue])
            if reference_points is None:
                delta = np.maximum(np.abs(xyz-center)-half, 0)
                distance2 = np.min(np.sum(delta**2, axis=1))
            else:
                delta = xyz[:, None, :]-np.asarray(reference_points)[None, :, :]
                distance2 = np.min(np.sum(delta**2, axis=2))
            if distance2 <= radius**2:
                retained.append(residue_id(chain, residue))
            else:
                remove.append(index)
        for index in reversed(remove):
            del chain[index]
    if not retained:
        raise ValueError('Empty receptor pocket; check box/reference ligand coordinate frame')
    pocket.write_pdb(str(path))
    return retained


def validate_receptor_pdbqt(structure, path):
    expected = [(a.element.atomic_number, np.asarray(list(a.pos))) for c in structure[0] for r in c for a in r]
    types = {'C': 6, 'A': 6, 'N': 7, 'NA': 7, 'OA': 8, 'O': 8, 'S': 16, 'SA': 16, 'P': 15,
             'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}
    actual = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        kind = line[77:].strip()
        if kind in {'H', 'HD', 'HS'}:
            continue
        if kind not in types:
            raise ValueError('Unsupported receptor PDBQT atom type: '+kind)
        actual.append((types[kind], np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    if len(actual) != len(expected):
        raise ValueError('Receptor preparation changed the heavy atom count')
    unused = set(range(len(actual))); maximum = 0.
    for number, xyz in expected:
        matches = [i for i in unused if actual[i][0] == number and np.max(np.abs(actual[i][1]-xyz)) <= .002]
        if len(matches) != 1:
            raise ValueError('Receptor preparation changed/ambiguously mapped heavy coordinates')
        index = matches[0]; unused.remove(index)
        maximum = max(maximum, float(np.linalg.norm(actual[index][1]-xyz)))
    return dict(heavy_atoms=len(expected), max_heavy_displacement_A=maximum)


def prepare_inputs(args, directory):
    directory = Path(directory); inputs = directory/'inputs'; prepared = directory/'prepared'
    inputs.mkdir(); prepared.mkdir()
    originals = dict(receptor=args.receptor, ligand=args.ligand)
    if args.reference_ligand is not None:
        originals['reference_ligand'] = args.reference_ligand
    hashes, copies = {}, {}
    for name, original in originals.items():
        original = Path(original).resolve(strict=True)
        copy = inputs/(name+original.suffix.lower())
        before = digest(original); shutil.copyfile(original, copy)
        if digest(copy) != before or digest(original) != before:
            raise ValueError('Input changed while copying: '+str(original))
        hashes[name] = dict(original=str(original), copied=str(copy), sha256=before)
        copies[name] = copy
    template = read_ligand(copies['ligand']); template.RemoveAllConformers()
    template_path = prepared/'ligand_template.sdf'
    with Chem.SDWriter(str(template_path)) as writer:
        writer.write(template)
    reference = reference_coordinates(copies['reference_ligand']) if 'reference_ligand' in copies else None
    box = resolve_box(args.center, args.size, reference, args.padding)
    receptor, receptor_audit = select_receptor(copies['receptor'], args.receptor_chains, args.model, args.remove_residues)
    residues = write_custom_pocket(receptor, prepared/'pocket.pdb', box, reference)
    receptor.write_pdb(str(prepared/'receptor.pdb'))
    prepare_receptor(prepared/'receptor.pdb', prepared/'receptor.pdbqt', str(args.obabel), prepared/'receptor_preparation.log')
    receptor_audit['pdbqt_coordinate_check'] = validate_receptor_pdbqt(receptor, prepared/'receptor.pdbqt')
    pose, mapping = prepare_start(template, prepared/'start.pdbqt', args.seed)
    with Chem.SDWriter(str(prepared/'ligand_start.sdf')) as writer:
        from vina_rtmscore.chemistry import molecule_from_pose
        writer.write(molecule_from_pose(template, pose, mapping, 'independent_start'))
    audit = dict(schema_version=1, inputs=hashes, box=box, receptor=receptor_audit,
        pocket=dict(strategy='distance_to_box' if reference is None else 'distance_to_reference_ligand',
                    radius_A=10., whole_residues=residues), index_map=mapping,
        ligand=dict(smiles=Chem.MolToSmiles(template), heavy_atoms=template.GetNumHeavyAtoms(),
                    formal_charge=Chem.GetFormalCharge(template), seed=args.seed,
                    template_representation='SDF-normalized bond endpoints; atom order/chemistry/stereo verified unchanged',
                    start='independent ETKDGv3; input coordinates discarded', protonation='input molecular species preserved'),
        prepared_hashes={p.name: digest(p) for p in prepared.iterdir() if p.is_file()})
    (directory/'preparation.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    return template, pose, mapping, audit
