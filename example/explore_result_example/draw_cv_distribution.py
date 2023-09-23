from openmmtools.multistate import MultiStateReporter
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("../..")
from openrid.colvar import calc_diherals
from openmm import unit
import numpy as np


# round_0_storage = "/home/gridsan/ywang3/Project/rid_openmm/ala2_out2/round_0/exploration/test.reporter.nc"
# round_1_storage = "/home/gridsan/ywang3/Project/rid_openmm/ala2_out2/round_1/exploration/test.reporter.nc"

storage_list = [
    f"/home/gridsan/ywang3/Project/rid_openmm/ala2_out3/round_{i}/exploration/test.reporter.nc" for i in range(4)
]

all_dih = []

for round_i, (reporter_storage) in enumerate(storage_list):
    reporter = MultiStateReporter(reporter_storage, open_mode='r')
    check_point_index = reporter.read_checkpoint_iterations()
    dihs = np.zeros((len(check_point_index), 2))
    for ii in check_point_index:
        cv_sampler = reporter.read_sampler_states(ii, analysis_particles_only=True)[0]
        pos = torch.from_numpy(cv_sampler.positions.value_in_unit(unit.nanometer))
        dih = calc_diherals(pos.reshape(-1, 4, 3))
        dihs[ii] = dih.detach().cpu().numpy()
    all_dih.append(dihs)

for cv_i in range(2):
    plt.figure()
    for round_i in range(len(all_dih)):
        plt.hist(all_dih[round_i][:,cv_i], bins=25, alpha=0.5, label=f"round_{round_i}_{cv_i}", density=True)
    plt.legend()
    plt.savefig(f"pics/cv_{cv_i}.png")
    plt.close()

plt.figure()
for round_i in range(len(all_dih)):
    plt.scatter(all_dih[round_i][:,0], all_dih[round_i][:,1], alpha=0.5, s=2, label=f"round_{round_i}")
plt.legend()
plt.xlabel("$\phi$")
plt.ylabel("$\psi$")
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)
plt.savefig(f"pics/cv.png")