from dataclasses import dataclass
from typing import List
from purkinje_learning.bo_utils import CriterionBO, TrainingDataSource, OptimizationMode, DeviceType, BOECGParameter, Prior

@dataclass
class PipelineConfig:
    patient_number: str
    N: int
    nIter: int
    criterion_bo: CriterionBO
    obtain_training_data: TrainingDataSource
    optimization_points: OptimizationMode
    device: DeviceType
    var_ecg_parameters: List[BOECGParameter]
    prior: Prior = Prior.UNIFORM
    plot: bool = True