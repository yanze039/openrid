import torch
import numpy as np
from abc import ABC, abstractmethod 


class CollectiveVariable(ABC):

    @abstractmethod
    def value_and_gradient(self, positions):
        """Calculate the value and gradient of the collective variable.
        Args:
            positions (torch.Tensor): [N_cv, N_atoms, 3]
                N_cv means how many collective variables are calculated. N_atoms means how many atoms 
                needed to calculate one collective variable. 3 means x, y, z coordinates.
        Returns:
            value (torch.Tensor): [N_cv]
            gradient (torch.Tensor): [N_cv, N_atoms, 3]
        """
        pass


class DihedralAngle(CollectiveVariable):

    def __init__(self, dihedral_index=None) -> None:
        super().__init__()
        self.dihedral_index = dihedral_index
    
    def value_and_gradient(self, positions):
        """Calculate the value and gradient of the collective variable.
        Args:
            positions (torch.Tensor): [N_cv, N_atoms, 3]
                N_cv means how many collective variables are calculated. N_atoms means how many atoms 
                needed to calculate one collective variable. 3 means x, y, z coordinates.
        Returns:
            value (torch.Tensor): [N_cv]
            gradient (torch.Tensor): [N_cv, N_atoms, 3]
        """
        p0 = positions[..., 0, :]
        p1 = positions[..., 1, :]
        p2 = positions[..., 2, :]
        p3 = positions[..., 3, :]

        F = p0 - p1
        G = p1 - p2
        H = p3 - p2

        A = torch.cross(F, G, dim=-1)  # [*, 3]
        B = torch.cross(H, G, dim=-1)

        G_norm = torch.norm(G, dim=-1, keepdim=True)  # [*, 1]
        A_sq = torch.sum(A**2, dim=-1, keepdim=True)  # [*, 1]
        B_sq = torch.sum(B**2, dim=-1, keepdim=True)  # [*, 1]
        F_dot_G = torch.sum(F * G, dim=-1, keepdim=True)  # [*, 1]
        H_dot_G = torch.sum(H * G, dim=-1, keepdim=True)  # [*, 1]
        aux = F_dot_G/(A_sq * G_norm) * A - H_dot_G/(B_sq * G_norm) * B  # [*, 3]
        
        A_x_B = torch.cross(A, B, dim=-1)  # [*, 3]
        torsion =  -torch.arctan2(
            (A_x_B * G).sum(-1) / G_norm.squeeze(-1),
            (A * B).sum(-1)

        )
        d_p0 =  - G_norm / A_sq * A
        d_p3 = G_norm / B_sq * B
        d_p1 = -d_p0 + aux
        d_p2 = - aux - d_p3

        return torsion, torch.stack([d_p0, d_p1, d_p2, d_p3], dim=-2)

        
    




@torch.jit.script
def calc_diherals(positions):
    """positions: [ *, 4, 3 ]"""
    """reference: 
        https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        Implementation of dihedral angle calculation based on Praxeolitic formula.
    """
    p0 = positions[..., 0, :]
    p1 = positions[..., 1, :]
    p2 = positions[..., 2, :]
    p3 = positions[..., 3, :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalize the edge vector
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    
    # get two in-plane vectors which are orthogonal to b1
    v = b0 - (b0*b1).sum(-1)[:,None] * b1  # b0 minus projection of b0 onto plane perpendicular to b1
    w = b2 - (b2*b1).sum(-1)[:,None] * b1  # b2 minus projection of b2 onto plane perpendicular to b1

    # angle between v and w in a plane is the torsion angle
    x = (v*w).sum(-1)
    y = (torch.cross(v, w, dim=-1)* b1).sum(-1)
    return torch.arctan2(y, x)


def calc_diherals_from_positions(positions):
    """positions: [ *, 4, 3 ]"""
    """reference: 
        https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        Implementation of dihedral angle calculation based on Praxeolitic formula.
    """
    p0 = positions[..., 0, :]
    p1 = positions[..., 1, :]
    p2 = positions[..., 2, :]
    p3 = positions[..., 3, :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalize the edge vector
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    
    # get two in-plane vectors which are orthogonal to b1
    v = b0 - (b0*b1).sum(-1)[:,None] * b1  # b0 minus projection of b0 onto plane perpendicular to b1
    w = b2 - (b2*b1).sum(-1)[:,None] * b1  # b2 minus projection of b2 onto plane perpendicular to b1

    # angle between v and w in a plane is the torsion angle
    x = (v*w).sum(-1)
    y = (np.cross(v, w, axis=-1)* b1).sum(-1)
    return np.arctan2(y, x)

@torch.jit.script
def calculate_dihedral_and_derivatives(positions):
    """positions: [ *, 4, 3 ]
       reference: 
            Blondel, A. and Karplus, M. (1996), New formulation for derivatives of torsion angles 
            and improper torsion angles in molecular mechanics: Elimination of singularities. J. 
            Comput. Chem., 17: 1132-1141. https://doi.org/10.1002/(SICI)1096-987X(19960715)17:9<1132::AID-JCC5>3.0.CO;2-T
    """
    p0 = positions[..., 0, :]
    p1 = positions[..., 1, :]
    p2 = positions[..., 2, :]
    p3 = positions[..., 3, :]

    F = p0 - p1
    G = p1 - p2
    H = p3 - p2

    A = torch.cross(F, G, dim=-1)  # [*, 3]
    B = torch.cross(H, G, dim=-1)

    G_norm = torch.norm(G, dim=-1, keepdim=True)  # [*, 1]
    A_sq = torch.sum(A**2, dim=-1, keepdim=True)  # [*, 1]
    B_sq = torch.sum(B**2, dim=-1, keepdim=True)  # [*, 1]
    F_dot_G = torch.sum(F * G, dim=-1, keepdim=True)  # [*, 1]
    H_dot_G = torch.sum(H * G, dim=-1, keepdim=True)  # [*, 1]
    aux = F_dot_G/(A_sq * G_norm) * A - H_dot_G/(B_sq * G_norm) * B  # [*, 3]
    
    A_x_B = torch.cross(A, B, dim=-1)  # [*, 3]
    torsion =  -torch.arctan2(
        (A_x_B * G).sum(-1) / G_norm.squeeze(-1),
        (A * B).sum(-1)

    )
    d_p0 =  - G_norm / A_sq * A
    d_p3 = G_norm / B_sq * B
    d_p1 = -d_p0 + aux
    d_p2 = - aux - d_p3

    return torsion, torch.stack([d_p0, d_p1, d_p2, d_p3], dim=-2)


if __name__ == "__main__":
    import numpy as np
    import MDAnalysis as mda
    from pathlib import Path
    from common import calc_dihedral

    data = Path("../data/")
    u = mda.Universe(data/"npt.gro")
    dihs = calc_dihedral("../data/npt.gro")
    dih_index = np.array([x[0] for x in dihs])
    dih_value = np.array([x[1] for x in dihs])
    model1 = DihedralAngle()
    protein = u.select_atoms("protein")
    dihedral_positions = protein.positions[dih_index.flatten()].reshape(-1, 4, 3)
    dihedrals = model1(torch.from_numpy(dihedral_positions))    
    dihedrals2, derivatives = calculate_dihedral_and_derivatives(torch.from_numpy(dihedral_positions))

    assert torch.allclose(dihedrals2, dihedrals)

    clac_dev = torch.func.jacrev(calc_diherals, argnums=0)
    dev = clac_dev(torch.from_numpy(dihedral_positions))
    for i in range(derivatives.shape[0]):
        assert (torch.allclose(derivatives[i], dev.sum(1)[i], atol=1e-5))

    