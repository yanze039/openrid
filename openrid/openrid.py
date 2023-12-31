import os
import logging
import json
import torch
import random
import shutil
import numpy as np
from pathlib import Path

import mpiplus
import openmmtools as mmtools
from openmm import unit

from openrid.colvar import calc_diherals_from_positions
from openrid.model import NeuralNetworkManager, DihedralBiasVmap
from openrid.Parser import YamlParser
from openrid.blocks import Exploration, Selector, RestrainedMDLabeler, ConcurrentExploration
import openrid.utils as utils

logger = logging.getLogger(__name__)


class ReinforcedDynamicsLoop:
    def __init__(
            self, 
            index,
            n_replicas,
            confs, 
            topology,
            cv_def,
            n_steps_exploration,
            traj_interval_exploration,
            info_interval_exploration,
            trajectory_checkpoint_interval_exploration,
            n_steps_labeling,
            checkpoint_interval_labeling,
            restraint_constant,
            temperature = 300,
            temperature_ladder = [],
            pressure = 1.0, 
            timestep = 4.0*unit.femtosecond, 
            timestep_nn = 4.0*unit.femtosecond,
            collision_rate = 1.0/unit.picoseconds,
            reporter = mmtools.multistate.MultiStateReporter, 
            reporter_name = "reporter.dcd", 
            reporter_checkpoint_interval_exploration = 2,
            n_remd_iters=800,
            n_steps_per_iter=500, 
            prior_model_path=None,
            platform="CUDA",
            trust_lvl_1=None,
            trust_lvl_2=None,
            init_trust_lvl_1=0.1,
            init_trust_lvl_2=0.2,
            n_cluaster_threshold = 8,
            n_cluster_lower_bound = 2,
            n_cluster_upper_bound = 4,
            distance_threshold = [0.1, 0.1, 0.1, 0.1],
            threshold_mode="search",
            pick_mode="all",
            n_model=4,
            epochs=100,
            model_features=[80,80,80,80],
            batch_size=128,
            dropout_rate=0.0,
            loop_output_dir=Path("loop"),
            resume=True,
            prior_data=None,
            exploration_mode="concurrent",
            shuffle_temperature=True,
        ) -> None:
        self.index = index
        self.n_replicas = n_replicas
        if isinstance(confs, str) or isinstance(confs, Path):
            self.confs = [confs]
        elif isinstance(confs, list):
            self.confs = confs
        else:
            raise TypeError("confs should be str or list of str")
        self.topology = topology
        self.temperature = temperature
        self.temperature_ladder = temperature_ladder
        if shuffle_temperature:
            random.shuffle(self.temperature_ladder)

        self.pressure = pressure
        self.timestep = timestep
        self.timestep_nn = timestep_nn
        self.n_remd_iters = n_remd_iters
        self.n_steps_per_iter = n_steps_per_iter
        self.reporter = reporter
        self.exploration_mode = exploration_mode
        if self.exploration_mode == "concurrent":
            self.checkpoint_interval_exploration = trajectory_checkpoint_interval_exploration
        else:
            self.checkpoint_interval_exploration = reporter_checkpoint_interval_exploration
        self.exploration_n_steps = n_steps_exploration
        self.exploration_traj_interval = traj_interval_exploration
        self.exploration_info_interval = info_interval_exploration

        self.platform = platform
        self.collision_rate = collision_rate
        self.cv_def = cv_def

        # selection
        self.trust_lvl_1 = trust_lvl_1
        self.trust_lvl_2 = trust_lvl_2
        self.n_cluster_lower_bound = n_cluster_lower_bound
        self.n_cluster_upper_bound = n_cluster_upper_bound
        self.distance_threshold = distance_threshold
        self.threshold_mode = threshold_mode
        self.n_cluaster_threshold = n_cluaster_threshold
        self.pick_mode = pick_mode
        self.trained_model_path = None
        
        # label
        self.n_steps_labeling = n_steps_labeling
        self.restraint_constant = restraint_constant
        self.checkpoint_interval_labeling = checkpoint_interval_labeling
        
        # network
        self.prior_model_path = prior_model_path
        self.n_model = n_model
        self.model_features = model_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate

        self.loop_output_dir = Path(loop_output_dir)
        self.exploration_dir = self.loop_output_dir / "exploration"
        if not os.path.exists(self.loop_output_dir):
            os.makedirs(self.loop_output_dir, exist_ok=True)
        if not os.path.exists(self.exploration_dir):
            os.makedirs(self.exploration_dir, exist_ok=True)
        self.reporter_name = reporter_name
        self.reporter_path = self.exploration_dir / self.reporter_name
        self.exploration_final_confs = [self.exploration_dir / f"conf_{state_index}.gro" for state_index in range(len(self.temperature_ladder))]
        
        self.exploration_trajs = [
            str(self.exploration_dir / (Path(reporter_name).stem + f"_{i}" + Path(self.reporter_name).suffix)) for i in range(n_replicas)
        ]
        self.checkpoint = self.loop_output_dir / "loop_checkpoint.json"
        self.selection_output_dir = self.loop_output_dir / "select"
        self.labeling_output_dir = self.loop_output_dir / "label"
        self.label_data_file_name = "data.raw.npy"
        self.data_dir = self.loop_output_dir / "data"
        self.data_file = self.data_dir / "data.npy"
        self.model_dir = self.loop_output_dir / "model"
        self.resume = resume
        self.prior_data = prior_data
        self.model_devi_threshold = init_trust_lvl_1
        self.init_trust_lvl_1 = init_trust_lvl_1
        self.init_trust_lvl_2 = init_trust_lvl_2
        self.trained_model_path = str(self.model_dir / f"model_best.pt")

        self.progress = {
            "exploration": False,
            "selection": False,
            "labeling": False,
            "training": False,
        }

    def write_checkpoint(self):
        mpicomm = mpiplus.get_mpicomm()
        try:
            mpicomm.barrier()
        except AttributeError:
            pass
        self._write_checkpoint()
    
    @mpiplus.on_single_node(0, sync_nodes=True)
    def _write_checkpoint(self):
        with open(self.checkpoint, "w") as f:
            json.dump(self.progress, f)
    
    @staticmethod
    def read_checkpoint(checkpoint):
        with open(checkpoint, "r") as f:
            return json.load(f)
    
    def recover_progress(self):
        with open(self.checkpoint, "r") as f:
            old_loop = json.load(f)
        self.progress = old_loop
    
    def run(self):
        
        if not os.path.exists(self.checkpoint):
            self.write_checkpoint()
        if self.resume:
            self.recover_progress()
        if not self.progress["exploration"]:
            self.run_exploration()
            self.progress["exploration"] = True
            self.write_checkpoint()
        if not self.progress["selection"]:
            self.run_selection()
            self.progress["selection"] = True
            distance_files = list(Path(self.selection_output_dir).glob("distance_*.txt"))
            for f in distance_files:
                state_idx = int(f.stem.split("_")[-1])
                with open(f, "r") as f:
                    self.distance_threshold[state_idx] = float(f.read().strip())
            self.write_checkpoint()
        if not self.progress["labeling"]:
            conformers = list(Path(self.loop_output_dir/"select").glob("*.gro"))
            self.run_labeling(conformers)
            self.progress["labeling"] = True
            self.write_checkpoint()
        if not self.progress["training"]:
            self.prepare_data()
            self.run_training()
            self.progress["training"] = True
            self.write_checkpoint()
        
        
    def run_exploration(self):
        if self.exploration_mode == "concurrent":
            self.exploration_step = ConcurrentExploration(
                topology=self.topology, 
                confs=self.confs, 
                n_replicas=self.n_replicas,
                model_path=self.prior_model_path,
                temperatures=self.temperature_ladder, 
                pressure=self.pressure, 
                timestep=self.timestep,
                reporter_name=self.reporter_name,
                cv_def=self.cv_def,
                n_steps=self.exploration_n_steps,
                traj_interval=self.exploration_traj_interval,
                info_interval=self.exploration_info_interval,
                chk_interval=self.checkpoint_interval_exploration,
                timestep_nn=self.timestep_nn,
                trust_lvl_1=self.trust_lvl_1, 
                trust_lvl_2=self.trust_lvl_2,
                groups=[(0,2), (1,1)], 
                collision_rate=self.collision_rate,
                output_dir=self.exploration_dir,
                resume=True
            )
        else:
            self.exploration_step = Exploration(
                        topology=self.topology, 
                        confs=self.confs, 
                        model_path=self.prior_model_path,
                        temperatures=self.temperature_ladder, 
                        pressure=self.pressure, 
                        timestep=self.timestep, 
                        n_remd_iters=self.n_remd_iters,
                        n_steps_per_iter=self.n_steps_per_iter, 
                        reporter=self.reporter, 
                        reporter_name=self.reporter_name, 
                        checkpoint_interval=self.checkpoint_interval_exploration,
                        timestep_nn=self.timestep_nn,
                        trust_lvl_1=self.trust_lvl_1, 
                        trust_lvl_2=self.trust_lvl_2,
                        platform=self.platform,
                        groups=[(0,2), (1,1)], 
                        collision_rate=self.collision_rate,
                        reassign_velocities=True,
                        output_dir=self.exploration_dir,
                        analysis_particle_indices=self.cv_def,
                        resume=self.resume,
                    )
        
        self.exploration_step.run()

    def run_selection(self):
        self.selection_step = Selector(
            topology=self.topology,
            cv_def=self.cv_def,
            name=self.index,
            model_path = self.prior_model_path,
            reporter = mmtools.multistate.MultiStateReporter, 
            n_cv = len(self.cv_def),
            model_devi_threshold=self.model_devi_threshold,
            n_cluaster_threshold = self.n_cluaster_threshold,
            n_cluster_lower_bound = self.n_cluster_lower_bound,
            n_cluster_upper_bound = self.n_cluster_upper_bound,
            distance_threshold = self.distance_threshold,
            threshold_mode=self.threshold_mode,
            pick_mode=self.pick_mode,
            output_dir = self.selection_output_dir,
            overwrite = True,
            groTopology = self.exploration_final_confs[0],
        )
        if self.exploration_mode == "concurrent":
            self.selection_step.select(self.exploration_trajs, mode="traj")
        else:
            self.selection_step.select(self.reporter_path, mode="reporter")
    
    def run_labeling(self, conformers):
        self.labeling_step = RestrainedMDLabeler(
            topology_file=self.topology, 
            conformers=conformers,
            cv_def=self.cv_def,
            restraint_constant=self.restraint_constant,
            cv_func=calc_diherals_from_positions,
            temperature=self.temperature, 
            n_steps=self.n_steps_labeling,
            traj_interval=self.checkpoint_interval_labeling,
            info_interval=self.n_steps_labeling/10,
            output_dir=self.labeling_output_dir,
            label_data_file_name=self.label_data_file_name,
            reset_box=True
        )
        self.labeling_step.run()
    
    @mpiplus.on_single_node(0, sync_nodes=True)
    def run_training(self):
        self.model_new = DihedralBiasVmap(
            colvar_idx=self.cv_def, 
            n_models=self.n_model, 
            n_cvs=len(self.cv_def), 
            dropout_rate=self.dropout_rate,
            features=self.model_features,
            e0=self.init_trust_lvl_1,
            e1=self.init_trust_lvl_2
        )
        logger.info(
            f"cv_def: {self.cv_def} \n" \
            f"n_model: {self.n_model} \n" \
            f"n_cv: {len(self.cv_def)}\n" \
            f"dropout_rate: {self.dropout_rate}\n" \
            f"features: {self.model_features}\n" \
        )
        training_step = NeuralNetworkManager(
            model = self.model_new,
            model_path = None,
            data_path = str(self.data_file),
            output_dir = self.model_dir,
            batch_size = self.batch_size,
            optimizer = torch.optim.Adam,
            learning_rate = 0.0008,
            decayRate = 0.96,
            epochs = self.epochs,
            loss_fn = torch.nn.MSELoss(),
            cv_num = len(self.cv_def),
            training_data_portion = 0.95,
        )
        training_step.train()
        
    @mpiplus.on_single_node(0, sync_nodes=True)
    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        
        if self.prior_data is None:
            shutil.copy(self.labeling_output_dir/ self.label_data_file_name, self.data_file)
        else:
            data_new = np.load(self.labeling_output_dir/ self.label_data_file_name)
            data_old = np.load(self.prior_data)
            assert data_new.shape[1] == data_old.shape[1] and len(data_new.shape) == len(data_old.shape)
            data = np.concatenate([data_old, data_new], axis=0)
            np.save(self.data_file, data)
        

class ReinforcedDynamics:
    def __init__(
            self,
            config = "rid.yaml"
    ) -> None:
        
        if config.endswith(".yaml") or config.endswith(".yml"):
            logger.info(f"Parsing YAML File {config} ...")
            self.parser = YamlParser(config)
        else:
            raise TypeError("config should be a yaml file")
        
        self.config = self.parser.parse()
        self.n_cycles = self.config["option"]["num_rid_cycles"]
        self.output_dir = Path(self.config["option"]["output_dir"])
        self.task_name = self.config["option"]["task_name"]
        self.storage_name = f"{self.task_name}.reporter.dcd"
        self.log_path = self.output_dir / f"{self.task_name}.log"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        utils.config_root_logger(self.config['verbose'], self.log_path)

        if self.config["option"]["platform"] == "auto" or self.config["option"]["platform"] is None:
            self.platform = mmtools.utils.get_fastest_platform()
        else:
            self.platform = mmtools.Platform.getPlatformByName(self.config["option"]["platform"])
        
        self.resume=self.config["option"]["resume"]
        self.checkpoint = self.output_dir / "checkpoint.json"
        self.cycle_index = 0
        self.record = {}
        if not os.path.exists(self.checkpoint):
            self.write_checkpoint()
        if self.resume:
            self.recover_cycle_index()
            self.recover_record()
        self.init_trust_lvl_1=self.config["option"]["trust_lvl_1"]
        self.init_trust_lvl_2=self.config["option"]["trust_lvl_2"]
        self.n_cluster_threshold = self.config["selection"]["n_cluster_threshold"]
        self.n_replicas = self.config["exploration"]["n_replicas"]
        
    def run(self):
        start = self.cycle_index
        logger.info("Total cycles is {}".format(self.n_cycles))
        for cycle in range(start, self.n_cycles):
            self.cycle_index = cycle
            logger.info(f" >>> Starting cycle {cycle}")
            self.record[str(cycle)] = {}
            self.write_checkpoint()
            mpicomm = mpiplus.get_mpicomm()
            try:
                mpicomm.barrier()
            except AttributeError:
                pass
            if cycle == 0:
                threshold_mode="search"
                pick_mode="all"
                distance_threshold = self.config["selection"]["cluster_threshold"]
                prior_data=self.config["option"]["initial_data"]
                confs = self.config["option"]["initial_conformers"]
            else:
                threshold_mode="fixed"
                pick_mode="model"
                distance_threshold = self.record[str(cycle-1)]["distance_threshold"]
                prior_data = self.record[str(cycle-1)]["data_file"]
                confs = self.record[str(cycle-1)]["final_confs"]
            if isinstance(distance_threshold, float):
                distance_threshold = [float(distance_threshold) for _ in range(self.n_replicas)]
            assert isinstance(distance_threshold, list)
            trust_lvl_1, trust_lvl_2 = self.get_trust_lvl()
            prior_model_path = self.get_model_path()
            loop_output_dir = self.output_dir / f"round_{cycle}"
            block = ReinforcedDynamicsLoop(
                index=cycle,
                confs=confs, 
                n_replicas=self.n_replicas,
                topology=self.config["option"]["topology_file"],
                temperature = self.config["option"]["temperature"]* unit.kelvin,
                temperature_ladder = [t * unit.kelvin for t in self.config["exploration"]["temperature_ladder"]],
                pressure = self.config["option"]["pressure"] * unit.atmospheres, 
                timestep = self.config["exploration"]["time_step"] * unit.picoseconds, 
                collision_rate=self.config["exploration"]["collision_rate"] / unit.picoseconds,
                n_remd_iters = self.config["exploration"]["n_remd_iters"],
                n_steps_per_iter = self.config["exploration"]["n_steps_per_iter"],
                n_steps_exploration = self.config["exploration"]["n_steps"],
                traj_interval_exploration = self.config["exploration"]["traj_interval"],
                info_interval_exploration = self.config["exploration"]["info_interval"],
                trajectory_checkpoint_interval_exploration = self.config["exploration"]["checkpoint_interval"],
                n_steps_labeling = self.config["labeling"]["n_steps"],
                checkpoint_interval_labeling = self.config["labeling"]["checkpoint_interval"],
                restraint_constant = self.config["labeling"]["kappa"] * unit.kilojoules_per_mole / (unit.radian**2),
                reporter = mmtools.multistate.MultiStateReporter, 
                reporter_name = str(self.storage_name),
                prior_model_path=prior_model_path,
                platform=self.platform,
                cv_def=self.config["collective_variables"]["cv_indices"],
                timestep_nn=self.config["exploration"]["time_step_nn"] * unit.picoseconds,
                trust_lvl_1=trust_lvl_1,
                trust_lvl_2=trust_lvl_2,
                init_trust_lvl_1= self.init_trust_lvl_1,
                init_trust_lvl_2= self.init_trust_lvl_2,
                n_cluaster_threshold = self.n_cluster_threshold,
                n_cluster_lower_bound = self.config["selection"]["n_cluster_lower"],
                n_cluster_upper_bound = self.config["selection"]["n_cluster_upper"],
                distance_threshold = distance_threshold,
                threshold_mode=threshold_mode, 
                pick_mode=pick_mode, 
                n_model=self.config["Train"]["n_models"],
                epochs=self.config["Train"]["epochs"],
                model_features=self.config["Train"]["neurons"],
                batch_size=self.config["Train"]["batch_size"],
                dropout_rate=self.config["Train"]["dropout_rate"],
                loop_output_dir=loop_output_dir,
                resume=self.resume,
                prior_data=prior_data
            )
            block.run()
            self.record[str(cycle)]["distance_threshold"] = block.distance_threshold
            self.record[str(cycle)]["trust_lvl_1"] = trust_lvl_1.tolist()
            self.record[str(cycle)]["trust_lvl_2"] = trust_lvl_2.tolist()
            self.record[str(cycle)]["model_path"] = str(block.trained_model_path)
            self.record[str(cycle)]["data_file"] = str(block.data_file)
            self.record[str(cycle)]["final_confs"] = [str(c) for c in block.exploration_final_confs]
            self.record[str(cycle)]["temperature_ladder"] = [float(t.value_in_unit(unit.kelvin)) for t in block.temperature_ladder]
            self.suggest_trust_lvl(block, cycle)
            self.write_checkpoint()
            mpicomm = mpiplus.get_mpicomm()
            try:
                mpicomm.barrier()
            except AttributeError:
                pass
    
    def get_model_path(self):
        if not hasattr(self, "cycle_index"):
            self.recover_cycle_index()
        if self.cycle_index == 0:
            return self.config["option"]["initial_model"]
        else:
            return self.record[str(self.cycle_index-1)]["model_path"]
    
    def get_trust_lvl(self):
        if not hasattr(self, "cycle_index"):
            self.recover_cycle_index()
        if self.cycle_index == 0:
            return np.ones(self.n_replicas)*self.init_trust_lvl_1, np.ones(self.n_replicas)*self.init_trust_lvl_2
        else:
            suggested_trust_lvl_1 = np.array(self.record[str(self.cycle_index-1)]["suggested_trust_lvl_1"])
            suggested_trust_lvl_2 = np.array(self.record[str(self.cycle_index-1)]["suggested_trust_lvl_2"])
            suggested_trust_lvl_1[suggested_trust_lvl_1 > 8 * self.init_trust_lvl_1] = self.init_trust_lvl_1
            suggested_trust_lvl_2[suggested_trust_lvl_1 > 8 * self.init_trust_lvl_1] = self.init_trust_lvl_2
            return suggested_trust_lvl_1, suggested_trust_lvl_2
    
    def suggest_trust_lvl(self, rid_loop, cycle):
        cluster_info = list(Path(rid_loop.selection_output_dir).glob(f"n_cluster_*.txt"))
        cluster_num_list = np.zeros(len(cluster_info), dtype=int)
        for cluster_f in cluster_info:
            cluster_idx = int(cluster_f.stem.split("_")[-1])
            with open(cluster_f, "r") as f:
                cluster_num_list[cluster_idx] = int(f.read().strip())
        original_trust_lvl_1 = np.array(self.record[str(cycle)]["trust_lvl_1"])
        original_trust_lvl_2 = np.array(self.record[str(cycle)]["trust_lvl_2"])
        original_trust_lvl_1[cluster_num_list<self.n_cluster_threshold] *= 1.5
        self.record[str(cycle)]["suggested_trust_lvl_1"] = original_trust_lvl_1.tolist()
        original_trust_lvl_2[cluster_num_list<self.n_cluster_threshold] = original_trust_lvl_1[cluster_num_list<self.n_cluster_threshold] + 1
        self.record[str(cycle)]["suggested_trust_lvl_2"] = original_trust_lvl_2.tolist()            

    @mpiplus.on_single_node(0, sync_nodes=True)
    def _write_checkpoint(self):
        with open(self.checkpoint, "w") as f:
            json.dump(self.record, f, indent=4)
    
    def write_checkpoint(self):
        mpicomm = mpiplus.get_mpicomm()
        try:
            mpicomm.barrier()
        except AttributeError:
            pass
        self._write_checkpoint()
    
    @staticmethod
    def read_checkpoint(checkpoint):
        with open(checkpoint, "r") as f:
            return json.load(f)
    
    def recover_cycle_index(self):
        with open(self.checkpoint, "r") as f:
            old_record = json.load(f)
        iter_idx = [int(s) for s in list(old_record.keys())]
        if len(iter_idx) == 0:
            self.cycle_index = 0
        else:
            self.cycle_index = max(iter_idx)
    
    def recover_record(self):
        with open(self.checkpoint, "r") as f:
            old_record = json.load(f)
        self.record = old_record

        
if __name__ == "__main__":
    rid = ReinforcedDynamics(config="./ala2.yaml")
    rid.run()


