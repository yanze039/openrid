import yaml
import copy
import warnings

CONFIG = {
    "option": {
        "initial_conformers": [],
        "topology_file": "../data/ala2.pdb",
        "task_name": "test",
        "num_rid_cycles": 20,
        "temperature": 300,
        "pressure": 1,
        "initial_models": None,
        "trust_lvl_1": 2,
        "trust_lvl_2": 3,
        "platform": "auto",
        "resume": True
    },
    "collective_variables": {
        "mode": "torsion",
        "cv_indices": [[0, 1, 2, 3]],
    },
    "exploration": {
        "n_remd_iters": 100,
        "n_steps_per_iter": 1000,
        "time_step": 0.002,
        "is_mts": True,
        "time_step_nn": 0.004,
        "collision_rate": 1,
        "checkpoint_interval": 10,
        "temperature_ladder": [300, 350, 400, 450, 500, 550, 600, 650, 700, 750],
    },
    "selection": {
        "cluster_threshold": 1,
        "n_cluster_lower": 12,
        "n_cluster_upper": 32,
        "n_cluster_threshold": 8,
    },
    "labeling": {
        "n_steps": 50000,
        "time_step": 0.002,
        "kappa": 500,
    },
    "Train": {
        "n_models": 4,
        "neurons": [50, 50, 50, 50],
        "resnet": True,
        "batch_size": 32,
        "epoches": 2000,
        "init_lr": 0.0008,
        "decay_steps": 120,
        "decay_rate": 0.96,
        "drop_out_rate": 0.1,
        "use_mix": False,
        "restart": False,
    },
    "verbose": False
}


class YamlParser:
    def __init__(self, path):
        self.path = path
        self.default_config = CONFIG
        
    def parse(self):
        if self.path is None:
            warnings.warn("No config file is provided, using default config")
            self.config = copy.deepcopy(self.default_config)
            return self.config
        
        with open(self.path, 'r') as fp:
            config = yaml.safe_load(fp)
        
        self.config = copy.deepcopy(self.default_config)
        for key, option in config.items():
            if key not in self.config:
                raise KeyError(f"Unknown key {key}")
            else:
                self.config[key].update(option)
        
        if self.config["option"]["initial_conformers"] is None:
            raise ValueError("No initial conformers are provided")
        
        if isinstance(self.config["option"]["initial_conformers"], str):
            self.config["option"]["initial_conformers"] = [self.config["option"]["initial_conformers"]]

        return self.config


        

        
        