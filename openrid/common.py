import MDAnalysis as mda
import numpy as np

def prep_dihedral(conf):
    u = mda.Universe(conf)
    dihedral_selection = []
    for res in u.residues:
        if res.resname == "SOL" or res.resname == "NA" or res.resname == "CL":
            continue
        if res.phi_selection() is not None:
            dihedral_selection.append([ii.index for ii in res.phi_selection()])
        if res.psi_selection() is not None:
            dihedral_selection.append([ii.index for ii in res.psi_selection()])
        
    return np.array(dihedral_selection)


def calc_dihedral(conf):
    u = mda.Universe(conf)
    dihedral_selection = []
    for res in u.residues:
        if res.resname == "SOL" or res.resname == "NA" or res.resname == "CL":
            continue
        if res.phi_selection() is not None:
            dihedral_selection.append([[ii.index for ii in res.phi_selection()], res.phi_selection().dihedral.value()])
        if res.psi_selection() is not None:
            dihedral_selection.append([[ii.index for ii in res.psi_selection()], res.psi_selection().dihedral.value()])
        
    return dihedral_selection


if __name__ == "__main__":
    conf = "../data/npt.gro"
    dihedral_selection = prep_dihedral(conf)
    print(dihedral_selection.shape)