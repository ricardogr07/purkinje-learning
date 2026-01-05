from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Protocol
import logging
from myocardial_mesh import MyocardialMesh

logger = logging.getLogger(__name__)

@dataclass
class ECGDataLoadResult:
    meshes_list: List[int]
    myocardial_mesh: MyocardialMesh
    patient_data_path: str


class ECGDataLoader(ABC):
    """
    Abstract base class for ECG data loaders.
    """
    def __init__(self, patient_id: str, device: str = "cpu"):
        self.patient_id = patient_id
        self.device = device

    @abstractmethod
    def validate_patient_id(self) -> None:
        pass

    @abstractmethod
    def get_mesh_paths(self) -> str:
        pass

    @abstractmethod
    def get_initial_nodes(self) -> List[int]:
        pass

    @abstractmethod
    def load(self) -> ECGDataLoadResult:
        pass
