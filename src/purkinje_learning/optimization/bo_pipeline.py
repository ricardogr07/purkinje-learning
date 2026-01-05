# src/purkinje_learning/optimization/bo_pipeline.py

from purkinje_learning import BO_PurkinjeTree, BO_ecg, OptimParam, Prior
from purkinje_learning.configs.pipeline_configs import PipelineConfig
from purkinje_learning.utils.var_ecg_parameters import (
    initial_values,
    var_ecg_parameters_list,
)
from purkinje_learning.data_loading import ECGDataLoadResult

import logging
from typing import List, Tuple
import numpy as onp

logger = logging.getLogger(__name__)

class BayesianOptimizationPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        data_result: ECGDataLoadResult,
        ecg_ground_truth: onp.ndarray
    ):
        self.config = config
        self.patient_prefix = data_result.patient_data_path
        self.meshes_list = data_result.meshes_list
        self.myocardium = data_result.myocardial_mesh
        self.ecg_ground_truth = ecg_ground_truth

    def run(self) -> Tuple[float, onp.ndarray, BO_ecg]:
        logger.info("Setting up Bayesian Optimization parameters...")

        var_params: List[OptimParam] = var_ecg_parameters_list(
            self.config.var_ecg_parameters,
            dim=2,
            prior=self.config.prior
        )
        logger.info(f"Variable ECG parameters for BO: {[p.parameter for p in var_params]}")

        config_bo = initial_values(var_params)
        logger.info(f"Initial values for BO_PurkinjeTreeConfig: {config_bo}")

        logger.info("Initializing BO_PurkinjeTree for optimization...")
        tree_bo = BO_PurkinjeTree(
            patient=self.patient_prefix,
            meshes_list=self.meshes_list,
            config=config_bo,
            myocardium=self.myocardium
        )

        logger.info("Wrapping BO_PurkinjeTree with BO_ecg...")
        bo_method = BO_ecg(bo_purkinje_tree=tree_bo, N=self.config.N)

        logger.info("Starting Bayesian Optimization with mse_jaxbo...")
        f, p_x_params, _ = bo_method.mse_jaxbo(
            ground_truth=self.ecg_ground_truth,
            variable_parameters=var_params
        )

        logger.debug(f"Objective function: {f}, Expected input shape: {getattr(p_x_params, 'shape', 'N/A')}")
        logger.info("Bayesian Optimization setup complete.")

        return f, p_x_params, bo_method
