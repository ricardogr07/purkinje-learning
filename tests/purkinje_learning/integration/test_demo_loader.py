import os
import pytest
from purkinje_learning import DemoECGDataLoader, ECGDataLoadResult


def test_demo_ecg_data_loader_outputs():
    loader = DemoECGDataLoader()
    result = loader.load()

    # Validate result structure
    assert isinstance(result, ECGDataLoadResult)
    assert isinstance(result.meshes_list, list)
    assert len(result.meshes_list) == 4
    assert all(isinstance(i, int) for i in result.meshes_list)
    assert hasattr(result.myocardial_mesh, "save")

    # Validate file existence
    assert os.path.isfile(os.path.join(result.patient_data_path, "crtdemo_mesh_oriented.vtk"))
    assert os.path.isfile(os.path.join(result.patient_data_path, "electrode_pos.pkl"))
    assert os.path.isfile(os.path.join(result.patient_data_path, "crtdemo_f0_oriented.vtk"))

    # Test saving to ensure mesh works
    tmp_path = "tests/integration"
    tmp_file = os.path.join(tmp_path, "tmp_output.vtu")

    # Ensure directory exists
    os.makedirs(tmp_path, exist_ok=True)

    try:
        result.myocardial_mesh.save(tmp_file)
        assert os.path.isfile(tmp_file)
    finally:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)
