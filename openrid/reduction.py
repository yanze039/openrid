import numpy as np
import torch
import math
import logging
import matplotlib.pyplot as plt


kbT = 8.314 * 500 / 1000
beta = 1.0 / kbT

logger = logging.getLogger(__name__)


class Walker(object):
    def __init__(self, n_walkers, model, cv_type, cv_lower=-math.pi, cv_upper=math.pi, _acp_ratio_lb=0.15, _acp_ratio_ub=0.75, device="cuda"):
        self.cv_type = cv_type
        self._full_dim = model.n_cvs
        self._num_walker = n_walkers
        
        self.model = model
        self._shape = (1, self._num_walker, self._full_dim)
        # if isinstance(cv_lower, float):
        #     cv_lower = torch.ones(self._full_dim) * cv_lower
        # if isinstance(cv_upper, float):
        #     cv_upper = torch.ones(self._full_dim) * cv_upper
        self.cv_upper = cv_upper
        self.cv_lower = cv_lower
        self._cv_range = (self.cv_upper - self.cv_lower)
        self._cv_shift = (self.cv_lower)
        self.use_pbc = True
        
        # absolute coordinate
        self._position = (torch.rand(size=self._shape)*(self._cv_range)) + (self._cv_shift)
        self._energy = torch.zeros([self._num_walker])
        self._sample_step = 20
        self._acp_ratio_lb = _acp_ratio_lb
        self._acp_ratio_ub = _acp_ratio_ub
        # self.max_scale = 5
        # self.min_scale = 0.01
        self._move_scale = 0.05
        self.inc_scale_fac = 1.25
        self._inner_steps=5
        self.device = device

        self._position = self._position.to(self.device).reshape(1, self._num_walker, self._full_dim).float()
        # self.cv_upper = self.cv_upper.to(self.device).reshape(1, 1, self._full_dim)
        # self.cv_lower = self.cv_lower.to(self.device).reshape(1, 1, self._full_dim)
        # self._cv_range = self._cv_range.to(self.device).reshape(1, 1, self._full_dim)
        # self._cv_shift = self._cv_shift.to(self.device).reshape(1, 1, self._full_dim)


    def step(self):
        acp_ratio_list = torch.zeros(self._inner_steps)
        self._energy = self.model.get_energy_from_torsion(self._position.float()).mean(0).detach().cpu()
        for ii in range(self._inner_steps):
            if self.use_pbc:
                position_new = self._position + torch.normal(mean=self._move_scale*torch.zeros_like(self._position),std=self._move_scale*torch.ones_like(self._position))
                position_new[position_new < self.cv_lower] = position_new[position_new < self.cv_lower] + self._cv_range
                position_new[position_new > self.cv_upper] = position_new[position_new > self.cv_upper] - self._cv_range
            # elif self.type == "distance":
            #     position_new = self._position + torch.normal(scale=self._move_scale*0.08, size=self._shape)
            #     indices = np.where(position_new < 0.1)
            #     position_new[indices]=0.13 + np.random.normal(scale=0.01, size=indices[0].shape)
            #     indices = np.where(position_new > 9.9)
            #     position_new[indices]=9.87 + np.random.normal(scale=0.01, size=indices[0].shape)
            else:
                raise ValueError("Undefined cv type, only support 'dih' and 'dis' type")
            energy_new = self.model.get_energy_from_torsion(position_new).mean(0).detach().cpu()
            # in case of overflow
            diff_energy = - (energy_new - self._energy) * beta
            # print(diff_energy)
            # exit(0)
            diff_energy[diff_energy > 0] = 0
            prob_ratio = torch.exp(diff_energy)
            idx = torch.rand(size=(self._num_walker,)) < prob_ratio.flatten()

            self._position[0, idx, :] = position_new[0, idx, :]
            self._energy[idx] = energy_new[idx]
            acp_ratio_list[ii] = torch.mean(idx.float())

        acp_ratio = torch.mean(acp_ratio_list)
        if acp_ratio > self._acp_ratio_ub:
            # move_scale is too small
            self._move_scale = self._move_scale * self.inc_scale_fac
            print(
                "Increase move_scale to %s due to high acceptance ratio: %f" % (
                    self._move_scale, acp_ratio))
            # print(self._position[:5, :, :])
        elif acp_ratio < self._acp_ratio_lb:
            # move_scale is too large
            self._move_scale = self._move_scale/self.inc_scale_fac
            print(
                "Decrease move_scale to %s due to low acceptance ratio: %f" % (
                    self._move_scale, acp_ratio))
        return self._position.detach().cpu(), self._energy
    


# project on all 1d points
def my_hist1d(pp, xx, delta, fd):
    my_hist = np.zeros((fd, len(xx)))
    for ii in range(pp.shape[0]):   ###trj_num
        for jj in range(fd):        ###cv_num
            my_hist[jj, int(pp[ii,jj]//delta)] += 1
    my_hist /= (pp.shape[0] * delta)
    return my_hist

# project on specific 2d point
def my_hist2d(pp, xx, yy, delta, cv1, cv2):
    my_hist = np.zeros((1, len(xx), len(yy)))
    for ii in range(pp.shape[0]):
        my_hist[0, int(pp[ii,cv1]//delta), int(pp[ii,cv2]//delta)] += 1
    my_hist /= (pp.shape[0] * delta * delta)
    return my_hist


if __name__ == "__main__":
    n_bins = 25
    grids = torch.linspace(-math.pi, math.pi, n_bins)
    steps = 1e6
    n_walkers = 6000
    mymodel = torch.load("/home/gridsan/ywang3/Project/rid_openmm/output/model/model_190_new.pt")
    mymodel.eval()
    mywalker = Walker(
        n_walkers=n_walkers, 
        model=mymodel,
        cv_type="torsion",
        cv_lower=-math.pi, 
        cv_upper=math.pi, 
        _acp_ratio_lb=0.30, 
        _acp_ratio_ub=0.70, 
        device="cuda"
    )
    print("warming up")
    for i in range(int(1000)):
        mywalker.step()
    import os
    hist = torch.zeros((18, n_bins-1))
    for i in range(int(steps)):
        position, energy = mywalker.step()
        for j in range(18):
            new_hist = torch.histogram(position.reshape(n_walkers, -1)[:,j], bins=grids, density=True)[0]
            hist[j] = (hist[j] * i + new_hist) / (i+1)
            if i % 1000 == 0:
                zz = -torch.log(hist[j]+1e-7).numpy()
                plt.figure()
                plt.plot(grids[:-1].numpy(), zz)
                if not os.path.exists(f"pics/{j}"):
                    os.makedirs(f"pics/{j}")
                plt.savefig(f"pics/{j}/hist_{i}.png")
                plt.close()
        if i % 1000 == 0:
            print(i)

