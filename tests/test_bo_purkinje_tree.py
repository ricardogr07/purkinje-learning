import pytest
import numpy as onp
from unittest.mock import patch, MagicMock

from bo_purkinje_tree import BO_PurkinjeTree, BO_PurkinjeTreeConfig, MeshSuffix


class MockParameters:
    def __init__(self):
        self.meshfile = ""
        self.init_node_id = 0
        self.second_node_id = 1
        self.init_length = 0.0
        self.length = 0.0
        self.w = 0.0
        self.l_segment = 0.0
        self.fascicles_length = []
        self.fascicles_angles = []
        self.branch_angle = 0.0
        self.N_it = 0
        self.save = False


class MockFractalTree:
    def __init__(self, params):
        self.params = params
        self.nodes_xyz = [[0.0, 0.0, 0.0]]
        self.connectivity = [[0, 0]]
        self.end_nodes = [0]

    def grow_tree(self):
        pass

    def save(self, path):
        return True


class MockPurkinjeTree:
    def __init__(self, xyz, conn, ends):
        self.xyz = onp.array(xyz)
        self.pmj = onp.array([0, 1], dtype=int)
        self.cv = 2.5

    def activate_fim(self, nodes, times):
        return onp.array([1.0, 2.0])


class MockMyocardialMesh:
    def __init__(self):
        self.xyz = onp.zeros((10, 3))

    def activate_fim(self, x0_xyz, x0_vals, return_only_pmjs=False):
        return onp.array(x0_vals) + 1.0

    def new_get_ecg(self, record_array=False):
        return onp.array([1.0, 2.0, 3.0])


@pytest.fixture
def config():
    return BO_PurkinjeTreeConfig(
        init_length=1.0,
        length=10.0,
        w=0.5,
        l_segment=0.1,
        fascicles_length=[3.0, 3.0],
        fascicles_angles=[0.1, 0.2],
        branch_angle=0.25,
        N_it=3,
        save_pmjs=False,
        kmax=3,
    )


@patch("bo_purkinje_tree.MyocardialMesh", autospec=True)
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
def test_initialize_calls_fractal_tree_correctly(
    mock_fractaltree,
    mock_parameters,
    mock_myocardium,
    config,
):
    tree = BO_PurkinjeTree(
        patient="patient01",
        meshes_list=[0, 1, 2, 3],
        config=config,
        myocardium=mock_myocardium,
    )

    assert isinstance(tree.LVfractaltree, MockFractalTree)
    assert isinstance(tree.RVfractaltree, MockFractalTree)
    assert tree.patient == "patient01"
    assert tree.init_length == config.init_length
    assert tree.branch_angle == config.branch_angle


@patch("vtk.vtkCellLocator")
@patch("pyvista.read")
@patch("purkinje_uv.fractal_tree_uv.Mesh")
@patch("purkinje_uv.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("purkinje_uv.Parameters", side_effect=MockParameters)
@patch("myocardial_mesh.MyocardialMesh", autospec=True)
def test_build_fractal_tree_parameters_sets_expected_values(
    mock_myocardium,
    mock_parameters,
    mock_fractaltree,
    mock_mesh,
    mock_pv_read,
    mock_vtk_locator,
    config,
):
    # Simulate pyvista.read returning a VTK-compatible mock
    mock_vtk_dataset = MagicMock()
    mock_vtk_dataset.points = onp.zeros((10, 3))
    mock_pv_read.return_value = mock_vtk_dataset

    # Simulate vtkCellLocator not failing
    mock_locator_instance = MagicMock()
    mock_vtk_locator.return_value = mock_locator_instance

    # Simulate mesh UV and connectivity
    mock_mesh_instance = MagicMock()
    mock_mesh_instance.uv = onp.zeros((10, 2))
    mock_mesh_instance.connectivity = onp.zeros((5, 3), dtype=int)
    mock_mesh_instance.compute_uvscaling.return_value = None
    mock_mesh.return_value = mock_mesh_instance

    # Run the class under test
    tree = BO_PurkinjeTree("patient01", [0, 1, 2, 3], config, mock_myocardium)

    # Test method
    result = tree._build_fractal_tree_parameters(MeshSuffix.LV, (0, 1))

    # Assertions
    assert result.meshfile == "patient01_LVendo_heart_cut.obj"
    assert result.init_node_id == 0
    assert result.second_node_id == 1
    assert result.init_length == config.init_length
    assert result.length == config.length
    assert result.w == config.w
    assert result.l_segment == config.l_segment
    assert result.fascicles_length == config.fascicles_length
    assert result.fascicles_angles == config.fascicles_angles
    assert result.branch_angle == config.branch_angle
    assert result.N_it == config.N_it
    assert result.save is False


@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
def test_apply_modifications_to_tree_modifies_expected_params(
    mock_parameters, mock_fractaltree, config
):
    # Setup
    tree = BO_PurkinjeTree("patient01", [0, 1, 2, 3], config, MagicMock())

    # Dummy trees
    lv_tree = MockFractalTree(MockParameters())
    rv_tree = MockFractalTree(MockParameters())

    # Patch param attributes
    lv_tree.params.branch_angle = 0.25
    rv_tree.params.branch_angle = 0.25
    lv_tree.params.N_it = 3
    rv_tree.params.N_it = 3

    modifications = {"branch_angle": [0.99, 0.88], "N_it": [10, 12]}

    tree._apply_modifications_to_tree(lv_tree, rv_tree, modifications, side="both")

    assert lv_tree.params.branch_angle == 0.99
    assert rv_tree.params.branch_angle == 0.88
    assert lv_tree.params.N_it == 10
    assert rv_tree.params.N_it == 12


@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
def test_apply_modifications_to_tree_both_valid(
    mock_parameters, mock_fractaltree, config
):
    tree = BO_PurkinjeTree("patient01", [0, 1, 2, 3], config, MagicMock())

    lv_tree = MockFractalTree(MockParameters())
    rv_tree = MockFractalTree(MockParameters())

    modifications = {"branch_angle": [0.99, 0.88], "N_it": [10, 12]}

    tree._apply_modifications_to_tree(lv_tree, rv_tree, modifications, side="both")

    assert lv_tree.params.branch_angle == 0.99
    assert rv_tree.params.branch_angle == 0.88
    assert lv_tree.params.N_it == 10
    assert rv_tree.params.N_it == 12


@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
def test_apply_modifications_to_tree_lv_only(mock_parameters, mock_fractaltree, config):
    tree = BO_PurkinjeTree("patient01", [0, 1, 2, 3], config, MagicMock())

    lv_tree = MockFractalTree(MockParameters())
    rv_tree = MockFractalTree(MockParameters())

    modifications = {"branch_angle": 1.23, "N_it": 9}

    tree._apply_modifications_to_tree(lv_tree, rv_tree, modifications, side="LV")

    assert lv_tree.params.branch_angle == 1.23
    assert lv_tree.params.N_it == 9

    # RV should remain default
    assert rv_tree.params.branch_angle == 0.0
    assert rv_tree.params.N_it == 0


@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
def test_apply_modifications_to_tree_both_scalar_raises(
    mock_parameters, mock_fractaltree, config
):
    tree = BO_PurkinjeTree("patient01", [0, 1, 2, 3], config, MagicMock())

    lv_tree = MockFractalTree(MockParameters())
    rv_tree = MockFractalTree(MockParameters())

    modifications = {"branch_angle": 0.99}

    with pytest.raises(
        ValueError,
        match=r"Expected list/tuple of length 2 for 'branch_angle' when side='both'",
    ):
        tree._apply_modifications_to_tree(lv_tree, rv_tree, modifications, side="both")


@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
def test_initialize_coupling_points_returns_expected_shapes(
    mock_params_cls, mock_tree_cls, config
):
    # Mock PMJ indices
    lv_pmj = onp.array([1, 2, 3])
    rv_pmj = onp.array([4, 5])

    # Mock trees
    mock_lv_tree = MockFractalTree(MockParameters())
    mock_lv_tree.pmj = lv_pmj
    mock_lv_tree.xyz = onp.random.rand(10, 3)

    mock_rv_tree = MockFractalTree(MockParameters())
    mock_rv_tree.pmj = rv_pmj
    mock_rv_tree.xyz = onp.random.rand(10, 3)

    # Myocardium mock (no PVCs)
    mock_myocardium = MagicMock()
    mock_myocardium.xyz = onp.empty((0, 3))

    # BO_PurkinjeTree instantiation
    tree = BO_PurkinjeTree("patient01", [0, 1, 2, 3], config, mock_myocardium)
    tree.save_pmjs = False

    result = tree._initialize_coupling_points(
        mock_lv_tree, mock_rv_tree, {}, modify=False
    )
    x0, x0_xyz, x0_vals, SIDE_LV, SIDE_RV, *_ = result

    # Assertions
    assert x0.shape[0] == 5
    assert x0_xyz.shape == (5, 3)
    assert x0_vals.shape == (5,)
    assert SIDE_LV.sum() == 3
    assert SIDE_RV.sum() == 2


@patch("bo_purkinje_tree.onp.savetxt")
@patch("bo_purkinje_tree.FractalTree", return_value=MockFractalTree(MockParameters()))
@patch("bo_purkinje_tree.Parameters", side_effect=MockParameters)
def test_initialize_coupling_points_with_modify_and_save(
    mock_params_cls,
    mock_tree_cls,
    mock_savetxt,
    config,
):
    # Mock PMJ indices
    lv_pmj = onp.array([1, 2, 3])
    rv_pmj = onp.array([4, 5])

    # Mock trees
    mock_lv_tree = MockFractalTree(MockParameters())
    mock_lv_tree.pmj = lv_pmj
    mock_lv_tree.xyz = onp.random.rand(10, 3)

    mock_rv_tree = MockFractalTree(MockParameters())
    mock_rv_tree.pmj = rv_pmj
    mock_rv_tree.xyz = onp.random.rand(10, 3)

    # Myocardium mock
    mock_myocardium = MagicMock()
    mock_myocardium.xyz = onp.empty((0, 3))

    # Instantiate tree with save_pmjs = True
    tree = BO_PurkinjeTree("data/patient01-case", [0, 1, 2, 3], config, mock_myocardium)
    tree.save_pmjs = True

    kwargs = {
        "cv": 3.3,
        "root_time": 5.0,
    }

    result = tree._initialize_coupling_points(
        mock_lv_tree, mock_rv_tree, kwargs, modify=True
    )
    x0, x0_xyz, x0_vals, SIDE_LV, SIDE_RV, LVroot, LVroot_time, RVroot, RVroot_time = (
        result
    )

    # Check conduction velocity was assigned
    assert mock_lv_tree.cv == 3.3
    assert mock_rv_tree.cv == 3.3

    # Check root times were computed properly
    assert LVroot == 0
    assert LVroot_time == -0.0  # min(0., 5.0) → 0 → -0.0
    assert RVroot == 0
    assert RVroot_time == 5.0

    # PMJs were attempted to be saved
    assert mock_savetxt.call_count == 2
    args_lv = mock_savetxt.call_args_list[0][0]
    args_rv = mock_savetxt.call_args_list[1][0]
    assert "LVpmj.txt" in args_lv[0]
    assert "RVpmj.txt" in args_rv[0]


def test_activate_purkinje_and_myo_updates_values_correctly():
    # Mock trees and myocardium
    LVtree = MagicMock()
    RVtree = MagicMock()
    myocardium = MagicMock()

    # Simulated input data
    x0 = onp.array([1, 2, 3, 4])
    x0_xyz = onp.random.rand(4, 3)
    x0_vals = onp.array([10.0, 20.0, 30.0, 40.0])
    LVroot, LVroot_time = 0, 5.0
    RVroot, RVroot_time = 0, 15.0
    SIDE_LV = onp.array([True, True, False, False])
    SIDE_RV = onp.array([False, False, True, True])

    # Simulate activation values from trees
    LVtree.activate_fim.return_value = onp.array([5.0, 10.0])  # less than original
    RVtree.activate_fim.return_value = onp.array(
        [35.0, 50.0]
    )  # greater and less than original

    # Simulate myocardial activation
    myocardium.activate_fim.return_value = onp.array([6.0, 8.0, 28.0, 45.0])

    # Instance of the class with mock myocardium
    bo_tree = BO_PurkinjeTree.__new__(BO_PurkinjeTree)
    bo_tree.myocardium = myocardium

    # Execute
    result = bo_tree._activate_purkinje_and_myo(
        LVtree,
        RVtree,
        x0.copy(),
        x0_xyz.copy(),
        x0_vals.copy(),
        LVroot,
        LVroot_time,
        RVroot,
        RVroot_time,
        SIDE_LV.copy(),
        SIDE_RV.copy(),
    )

    # Call verifications
    LVtree.activate_fim.assert_called_once()
    RVtree.activate_fim.assert_called_once()
    myocardium.activate_fim.assert_called_once()

    # Check expected result
    expected = onp.array(
        [
            min(5.0, 6.0),  # LV
            min(10.0, 8.0),  # LV
            min(35.0, 28.0),  # RV
            min(50.0, 45.0),  # RV
        ]
    )
    onp.testing.assert_array_equal(result, expected)


@patch("bo_purkinje_tree.PurkinjeTree")
def test_run_ECG_executes_and_returns_expected_outputs(mock_purkinje_tree):
    # Instance of BO_PurkinjeTree without __init__
    bo = BO_PurkinjeTree.__new__(BO_PurkinjeTree)

    # Simulated attributes
    bo.kmax = 3
    bo.LVfractaltree = MagicMock()
    bo.RVfractaltree = MagicMock()
    bo._apply_modifications_to_tree = MagicMock()
    bo._initialize_coupling_points = MagicMock()
    bo._activate_purkinje_and_myo = MagicMock()
    bo.myocardium = MagicMock()

    # Simulated grow_tree methods
    bo.LVfractaltree.grow_tree = MagicMock()
    bo.RVfractaltree.grow_tree = MagicMock()

    # Simulated PurkinjeTree constructor
    LVtree_mock = MagicMock()
    RVtree_mock = MagicMock()
    mock_purkinje_tree.side_effect = [LVtree_mock, RVtree_mock]

    # Simulated input data
    x0 = onp.array([0, 1])
    x0_xyz = onp.random.rand(2, 3)
    x0_vals_initial = onp.array([10.0, 20.0])
    SIDE_LV = onp.array([True, False])
    SIDE_RV = onp.array([False, True])
    LVroot, LVroot_time = 0, 5.0
    RVroot, RVroot_time = 1, 15.0

    # Mock activation
    x0_vals_step1 = onp.array([9.0, 19.0])
    x0_vals_step2 = onp.array([8.5, 18.0])
    bo._initialize_coupling_points.return_value = (
        x0,
        x0_xyz,
        x0_vals_initial.copy(),
        SIDE_LV,
        SIDE_RV,
        LVroot,
        LVroot_time,
        RVroot,
        RVroot_time,
    )
    bo._activate_purkinje_and_myo.side_effect = [x0_vals_step1, x0_vals_step2]

    # ECG mock with convergence in the second iteration
    ecg1 = onp.array([1.0, 2.0, 3.0])
    ecg2 = onp.array([1.001, 2.001, 3.001])
    bo.myocardium.new_get_ecg.side_effect = [
        ecg1,
        ecg2,
        ecg2,
    ]  # 2 in the loop + 1 final

    # Execute
    ecg_out, lv_out, rv_out = bo.run_ECG(modify=False)

    # Verifications
    assert onp.allclose(ecg_out, ecg2)
    assert lv_out == LVtree_mock
    assert rv_out == RVtree_mock
    assert bo._initialize_coupling_points.called
    assert bo._activate_purkinje_and_myo.call_count == 2  # stopped due to convergence
    assert bo.myocardium.new_get_ecg.call_count == 3


def test_save_fractaltrees_saves_successfully():
    # Mock fractal tree objects with .save() method
    mock_lv_tree = MagicMock()
    mock_rv_tree = MagicMock()

    tree = BO_PurkinjeTree.__new__(BO_PurkinjeTree)
    tree.LVfractaltree = mock_lv_tree
    tree.RVfractaltree = mock_rv_tree

    success = tree.save_fractaltrees("LVtree.obj", "RVtree.obj")

    mock_lv_tree.save.assert_called_once_with("LVtree.obj")
    mock_rv_tree.save.assert_called_once_with("RVtree.obj")
    assert success is True
