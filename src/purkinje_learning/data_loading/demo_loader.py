from purkinje_learning.data_loading.base_loader import ECGDataLoader, ECGDataLoadResult
from myocardial_mesh import MyocardialMesh
from typing import List
import logging

logger = logging.getLogger(__name__)


class DemoECGDataLoader(ECGDataLoader):
    def __init__(self, device: str = "cpu"):
        super().__init__(patient_id="demo", device=device)
        self.validate_patient_id()

    def validate_patient_id(self) -> None:
        if self.patient_id != "demo":
            raise ValueError("Only 'demo' patient is supported.")

    def get_mesh_paths(self) -> str:
        return "data/crtdemo"

    def get_initial_nodes(self) -> List[int]:
        # Hardcoded for demo patient
        # These are node indices (of the LV and RV endocardial meshes) that determine the direction of the
        # initial branch of the Purkinje Tree
        # Here, 388 and 412 are nodes of the LV endocardial mesh and
        #       198 and 186 are nodes of the RV endocardial mesh
        return [388, 412, 198, 186]

    def load(self) -> ECGDataLoadResult:
        base_path = self.get_mesh_paths()

        mesh = MyocardialMesh(
            mesh_path=f"{base_path}/crtdemo_mesh_oriented.vtk",
            fibers_path=f"{base_path}/crtdemo_f0_oriented.vtk",
            electrodes_position=f"{base_path}/electrode_pos.pkl",
            device=self.device,
        )

        return ECGDataLoadResult(
            meshes_list=self.get_initial_nodes(),
            myocardial_mesh=mesh,
            patient_data_path=base_path,
        )
