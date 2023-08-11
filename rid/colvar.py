import torch
import numpy as np
from pathlib import Path
# import MDAnalysis as mda
# from MDAnalysis.analysis.dihedrals import Dihedral

class DihedralAngle(torch.nn.Module):

    def __init__(self):
        """Dihedral angle between four particles."""
        """reference: 
        https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        Implementation of dihedral angle calculation based on Praxeolitic formula.
        """
        super().__init__()


    def forward(self, positions):
        """positions: [ *, 4, 3 ]"""
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
        return torch.arctan2(y, x).requires_grad_(True)

 