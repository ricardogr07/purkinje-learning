import os
import pickle
import logging
import numpy as onp
from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt

from purkinje_learning import BO_PurkinjeTreeConfig, BO_PurkinjeTree, BO_ecg
from purkinje_learning.data_loading import ECGDataLoadResult

logger = logging.getLogger(__name__)

onp.random.seed(1234)

class GroundTruthGenerator:
    def __init__(
        self,
        data_result: ECGDataLoadResult,
        patient_number: int,
        output_root: str = "./output",
        var_param_str: str = "init_length_fascicles_length_fascicles_angles_root_time_cv",
    ):
        self.patient_data_path = data_result.patient_data_path
        self.meshes_list = data_result.meshes_list
        self.myocardium = data_result.myocardial_mesh
        self.patient_number = patient_number
        self.output_dir = os.path.join(output_root, f"patient{patient_number}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.var_param_str = var_param_str

    def generate(self) -> tuple[onp.ndarray, dict]:
        """
        Generates the ground-truth ECG and Purkinje structures based on known good parameters.

        It performs the following steps:
        - Loads the fixed configuration for reference tree
        - Initializes the Purkinje and BO models
        - Loads precomputed parameters (argmin of past BO run)
        - Simulates ECG with those parameters
        - Saves ECG, myocardium and Purkinje trees

        Returns
        -------
        tuple:
            - ecg_pat_array : np.ndarray
                The trimmed ECG signal (QRS segment).
            - var_params_true : dict
                Dictionary of the ground-truth parameter values.
        """
        logger.info("Starting ground truth generation...")

        # Load static config
        true_parameters_config = self._get_reference_tree_config()
        # Build full prefix for patient data
        patient_prefix = os.path.join(self.patient_data_path, "crtdemo")

        # Initialize Purkinje and BO models
        _, bo_model = self._initialize_reference_model(
            patient_prefix, true_parameters_config
        )
        
        # Construct input paths
        X_path = os.path.join(
            self.output_dir,
            f"data_N_250_nIter_300_criterionEI_{self.var_param_str}_X.npy"
        )
        y_path = os.path.join(
            self.output_dir,
            f"data_N_250_nIter_300_criterionEI_{self.var_param_str}_y.npy"
        )

        # Load parameter values from argmin
        var_params_true = self._load_ground_truth_parameters(X_path, y_path)

        # Run ECG simulation
        ecg_true, LVtree_true, RVtree_true = self._simulate_ecg(
            model=bo_model,
            parameters=var_params_true,
            side="both"
        )

        # Save all results
        self._save_reference_outputs(
            ecg=ecg_true,
            LVtree=LVtree_true,
            RVtree=RVtree_true
        )

        qrs_in, qrs_fin = 0, len(ecg_true)
        ecg_pat_array = ecg_true[qrs_in:qrs_fin]

        return ecg_pat_array, var_params_true

    def _get_reference_tree_config(self) -> BO_PurkinjeTreeConfig:
        return BO_PurkinjeTreeConfig(
            init_length=0.0,
            length=8.0,
            w=0.1,
            l_segment=1.0,
            fascicles_length=[0.0, 0.0],
            fascicles_angles=[0.0, 0.0],
            branch_angle=0.15,
            N_it=20,
        )

    def _initialize_reference_model(
        self,
        patient_prefix: str,
        config: BO_PurkinjeTreeConfig
    ) -> tuple[BO_PurkinjeTree, BO_ecg]:
        logger.info("Initializing reference Purkinje tree and BO model...")
        purkinje_tree = BO_PurkinjeTree(
            patient=patient_prefix,
            meshes_list=self.meshes_list,
            config=config,
            myocardium=self.myocardium
        )
        bo_model = BO_ecg(bo_purkinje_tree=purkinje_tree)
        return purkinje_tree, bo_model

    def _load_ground_truth_parameters(self, X_path: str, y_path: str) -> dict:
        logger.info("Loading results for ground truth...")
        X_read = onp.load(X_path)
        y_read = onp.load(y_path)

        X_min = X_read[onp.argmin(y_read)]

        var_params_true = {
            "init_length": [X_min[0], X_min[1]],
            "fascicles_length": [
                [0.5 * X_min[2], 0.5 * X_min[3]],
                [0.5 * X_min[4], 0.5 * X_min[5]],
            ],
            "fascicles_angles": [
                [0.1 * X_min[6], 0.1 * X_min[7]],
                [0.1 * X_min[8], 0.1 * X_min[9]],
            ],
            "root_time": X_min[10],
            "cv": X_min[11],
        }

        logger.info("Ground truth parameters:")
        for k, v in var_params_true.items():
            logger.info(f"{k}: {v}")

        return var_params_true

    def _simulate_ecg(
        self,
        model: BO_ecg,
        parameters: dict,
        side: str = "both"
    ) -> tuple[onp.ndarray, object, object]:
        logger.info("Simulating reference ECG with true parameters...")
        ecg, LVtree, RVtree = model.bo_purkinje_tree.run_ECG(
            n_sim=0, modify=True, side=side, **parameters
        )
        return ecg, LVtree, RVtree

    def _save_reference_outputs(
        self,
        ecg: onp.ndarray,
        LVtree,
        RVtree
    ) -> None:
        logger.info("Saving reference ECG and geometries...")
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "True_ecg"), "wb") as f:
            pickle.dump(ecg, f)

        self.myocardium.save(os.path.join(self.output_dir, "True_endo.vtu"))
        LVtree.save(os.path.join(self.output_dir, "True_LVtree.vtu"))
        RVtree.save(os.path.join(self.output_dir, "True_RVtree.vtu"))

    def plot_ecg(self, ecg_array: onp.ndarray, title: str = "Reference ECG", save_path: Optional[Union[str, Path]] = None) -> None:
        logger.info("Plotting reference ECG...")
        fig, axs = plt.subplots(3, 4, figsize=(10, 13), dpi=120, sharex=True, sharey=True)
        for ax, lead in zip(axs.ravel(), ecg_array.dtype.names):
            ax.plot(ecg_array[lead])
            ax.grid()
            ax.set_title(lead)
        fig.suptitle(title)
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            fig.savefig(save_path)
            logger.info(f"Saved ECG plot to {save_path}")

        plt.show()

