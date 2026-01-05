import os
import shutil
from pathlib import Path
import pytest
import numpy as onp

from purkinje_learning.data_loading import DemoECGDataLoader
from purkinje_learning.data_loading.ground_truth_generator import GroundTruthGenerator

# Seed numpy for reproducibility
onp.random.seed(1234)

@pytest.mark.integration
def test_ground_truth_generator_pipeline(tmp_path):
    """
    Full integration test for GroundTruthGenerator using demo data.
    """

    # Load real demo data
    loader = DemoECGDataLoader()
    data_result = loader.load()

    # Copiar archivos de output reales a tmp_path
    real_output_dir = Path("output") / "patientdemo"
    test_output_dir = tmp_path / "patient1"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "data_N_250_nIter_300_criterionEI_init_length_fascicles_length_fascicles_angles_root_time_cv_X.npy",
        "data_N_250_nIter_300_criterionEI_init_length_fascicles_length_fascicles_angles_root_time_cv_y.npy"
    ]
    missing = [fname for fname in required_files if not (real_output_dir / fname).is_file()]
    if missing:
        pytest.skip(f"Missing ground truth files in {real_output_dir}: {missing}")

    for fname in required_files:
        shutil.copy(real_output_dir / fname, test_output_dir)

    # Use a temporary output directory to avoid writing to real output/
    generator = GroundTruthGenerator(
        data_result=data_result,
        patient_number=1,
        output_root=tmp_path
    )

    # Run generation
    ecg_array, params = generator.generate()

    # Check ECG is not empty and has expected dtype
    assert isinstance(ecg_array, onp.ndarray)
    assert ecg_array.shape[0] > 0
    assert ecg_array.dtype.names is not None

    expected_leads = {"E1", "E2", "E3", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}
    actual_leads = set(ecg_array.dtype.names)

    missing = expected_leads - actual_leads
    assert not missing, f"Missing leads: {missing}"

    # Check all expected parameter keys exist
    expected_keys = ["init_length", "fascicles_length", "fascicles_angles", "root_time", "cv"]
    for key in expected_keys:
        assert key in params

    # Check files were created
    output_dir = Path(generator.output_dir)
    assert (output_dir / "True_ecg").exists()
    assert (output_dir / "True_endo.vtu").exists()
    assert (output_dir / "True_LVtree.vtu").exists()
    assert (output_dir / "True_RVtree.vtu").exists()

    # Plot and save
    plot_path = tmp_path / "reference_ecg.png"
    generator.plot_ecg(ecg_array, save_path=plot_path)
    assert plot_path.exists()
