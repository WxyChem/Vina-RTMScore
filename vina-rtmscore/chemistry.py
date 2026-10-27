"""Recover coordinates using the explicit ligand atom map."""
def molecule_from_pose(template,pose,index_map,name):
    from rdkit import Chem
    # Existing Meeko map: one-based chemical atom index -> PDBQT serial.
    mapping={int(v):int(k)-1 for k,v in index_map.items()}
    if set(mapping.values())!=set(range(template.GetNumAtoms())) or len(mapping)!=template.GetNumAtoms():
        raise ValueError('Incomplete heavy-atom map')
    positions={s:p for s,p in zip(pose.serials,pose.coords)}
    if not set(mapping).issubset(positions):
        raise ValueError('Missing pose atom')
    mol=Chem.Mol(template);mol.RemoveAllConformers()
    conf=Chem.Conformer(mol.GetNumAtoms())
    for serial,index in mapping.items(): conf.SetAtomPosition(index,positions[serial])
    mol.AddConformer(conf);mol.SetProp('_Name',name)
    return mol
