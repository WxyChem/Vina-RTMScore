import os
import re
import subprocess

import prody as pr


def write_file(output_file, outline):
    with open(output_file, "w") as buffer:
        buffer.write(outline)


def lig_rename(infile, outfile):
    """
    Some peptides may impede the generation of pocket,
    so rename the ligand residue name to LIG first.
    """
    with open(infile, "r") as f:
        lines = f.readlines()

    newlines = []

    for line in lines:
        if re.search(r"^HETATM|^ATOM", line):
            newlines.append(line[:17] + "LIG" + line[20:])
        else:
            newlines.append(line)

    write_file(outfile, "".join(newlines))


def check_mol(infile, outfile):
    """
    Some metals may have the same ID as ligand,
    thus making ligand included in the pocket.
    """
    os.system("cat %s | sed '/LIG/d' > %s" % (infile, outfile))


def extract_pocket(
    protpath,
    ligpath,
    cutoff=5.0,
    protname=None,
    ligname=None,
    workdir="."
):
    """
    protpath: path of protein file (.pdb)
    ligpath: path of ligand file (.sdf | .mol2 | .pdb)
    cutoff: distance range within ligand to determine pocket
    protname: protein name
    ligname: ligand name
    workdir: working directory
    """

    if protname is None:
        protname = os.path.basename(protpath).split(".")[0]

    if ligname is None:
        ligname = os.path.basename(ligpath).split(".")[0]

    # ==========================================================
    # Convert ligand to PDB using Open Babel command line
    # ==========================================================

    if not re.search(r"\.pdb$", ligpath, re.IGNORECASE):

        output_ligand = os.path.join(
            workdir,
            "%s.pdb" % ligname
        )

        obabel = "/home/mol/miniconda3/envs/karmadock/bin/obabel"

        cmd = [
            obabel,
            ligpath,
            "-O",
            output_ligand
        ]

        print("=" * 60)
        print("Open Babel conversion")
        print("Input :", ligpath)
        print("Output:", output_ligand)
        print("Command:", " ".join(cmd))
        print("=" * 60)
        
        env = os.environ.copy()

        env["BABEL_LIBDIR"] = \
            "/home/mol/miniconda3/envs/karmadock/lib/openbabel/3.1.0"
        
        env["BABEL_DATADIR"] = \
            "/home/mol/miniconda3/envs/karmadock/share/openbabel"
        
        print("========================================")
        print("BABEL_LIBDIR =", env["BABEL_LIBDIR"])
        print("BABEL_DATADIR =", env["BABEL_DATADIR"])
        print("========================================")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        print("Open Babel stdout:")
        print(result.stdout)

        print("Open Babel stderr:")
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                "Open Babel conversion failed!\n"
                "Command: %s\n"
                "stdout:\n%s\n"
                "stderr:\n%s"
                % (
                    " ".join(cmd),
                    result.stdout,
                    result.stderr
                )
            )

        if not os.path.isfile(output_ligand):
            raise RuntimeError(
                "Open Babel finished successfully, "
                "but output file was not generated:\n%s"
                % output_ligand
            )

        print("Successfully generated:", output_ligand)

    else:
        output_ligand = ligpath

    # ==========================================================
    # Parse protein
    # ==========================================================

    xprot = pr.parsePDB(protpath)

    if xprot is None:
        raise RuntimeError(
            "ProDy failed to parse protein: %s" % protpath
        )

    # ==========================================================
    # Rename ligand residue to LIG
    # ==========================================================

    ligand_pdb = os.path.join(
        workdir,
        "%s.pdb" % ligname
    )

    ligand_pdb2 = os.path.join(
        workdir,
        "%s2.pdb" % ligname
    )

    lig_rename(
        ligand_pdb,
        ligand_pdb2
    )

    os.remove(ligand_pdb)

    os.rename(
        ligand_pdb2,
        ligand_pdb
    )

    # ==========================================================
    # Parse ligand
    # ==========================================================

    xlig = pr.parsePDB(ligand_pdb)

    if xlig is None:
        raise RuntimeError(
            "ProDy failed to parse ligand: %s" % ligand_pdb
        )

    if len(xlig) == 0:
        raise RuntimeError(
            "Ligand contains no atoms: %s" % ligand_pdb
        )

    lresname = xlig.getResnames()[0]

    print("Ligand residue name:", lresname)
    print("Ligand atoms:", len(xlig))
    print("Protein atoms:", len(xprot))

    # ==========================================================
    # Combine ligand and protein
    # ==========================================================

    xcom = xlig + xprot

    # ==========================================================
    # Select pocket
    # ==========================================================

    selection = (
        "same residue as exwithin %s of resname %s"
        % (cutoff, lresname)
    )

    print("Pocket selection:", selection)

    ret = xcom.select(selection)

    if ret is None:
        raise RuntimeError(
            "Failed to generate pocket selection."
        )

    # ==========================================================
    # Write pocket
    # ==========================================================

    pocket_temp = os.path.join(
        workdir,
        "%s_pocket_%s_temp.pdb" % (
            protname,
            cutoff
        )
    )

    pocket = os.path.join(
        workdir,
        "%s_pocket_%s.pdb" % (
            protname,
            cutoff
        )
    )

    pr.writePDB(
        pocket_temp,
        ret
    )

    check_mol(
        pocket_temp,
        pocket
    )

    os.remove(pocket_temp)

    print("Pocket generated:", pocket)

    return pocket
    