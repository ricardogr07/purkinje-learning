import pytest
import numpy as np
from bo_ecg import BO_ecg, OptimParam, PriorType
from unittest.mock import patch

@pytest.fixture
def dummy_bo():
    return BO_ecg(bo_purkinje_tree=object())

def test_set_dictionary_variables_scalar_and_vector(dummy_bo):

    var_params = [
        OptimParam(
            parameter="cv",
            lower=np.array([2.0]),
            upper=np.array([4.0]),
            prior=PriorType.UNIFORM
        ),
        OptimParam(
            parameter="fascicles_angles",
            lower=np.array([-0.5, -0.5, -0.5, -0.5]),
            upper=np.array([0.5, 0.5, 0.5, 0.5]),
            prior=PriorType.UNIFORM
        ),
    ]

    # Vector plano con valores
    x_values = np.array([3.0, 0.1, 0.2, 0.3, 0.4])

    # Ejecutar función
    result = dummy_bo.set_dictionary_variables(var_params, x_values)

    # Verificaciones
    assert isinstance(result, dict)
    assert "cv" in result and result["cv"] == 3.0
    assert "fascicles_angles" in result
    assert result["fascicles_angles"] == [[0.1, 0.2], [0.3, 0.4]]

def test_extract_overlapping_section_zero_delay():
    bo = BO_ecg(bo_purkinje_tree=None)

    ground_truth = np.array([1, 2, 3, 4, 5])
    predicted = np.array([10, 20, 30, 40, 50])
    delay = 0

    gt_section, pred_section = bo._extract_overlapping_section(ground_truth, predicted, delay)

    assert np.array_equal(gt_section, ground_truth)
    assert np.array_equal(pred_section, predicted)

def test_extract_overlapping_section_positive_delay():
    bo = BO_ecg(bo_purkinje_tree=None)

    ground_truth = np.array([1, 2, 3, 4, 5])
    predicted = np.array([10, 20, 30, 40, 50])
    delay = 2  # Predicción retrasada

    gt_section, pred_section = bo._extract_overlapping_section(ground_truth, predicted, delay)

    assert np.array_equal(gt_section, ground_truth[2:])
    assert np.array_equal(pred_section, predicted[:3])

def test_extract_overlapping_section_negative_delay():
    bo = BO_ecg(bo_purkinje_tree=None)

    ground_truth = np.array([1, 2, 3, 4, 5])
    predicted = np.array([10, 20, 30, 40, 50])
    delay = -2  # Predicción adelantada

    gt_section, pred_section = bo._extract_overlapping_section(ground_truth, predicted, delay)

    assert np.array_equal(gt_section, ground_truth[:3])
    assert np.array_equal(pred_section, predicted[2:])


def test_extract_overlapping_section_invalid_overlap():
    bo = BO_ecg(bo_purkinje_tree=None)

    ground_truth = np.array([1, 2, 3])
    predicted = np.array([10, 20, 30])
    delay = 5  # Demasiado retrasado, no se superpone nada

    with pytest.raises(ValueError, match="No overlapping region found"):
        bo._extract_overlapping_section(ground_truth, predicted, delay)


def test_loss_direct_perfect_match():
    bo = BO_ecg(bo_purkinje_tree=None)

    dtype = [("I", float), ("II", float)]
    ground_truth = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)
    predicted = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)

    loss = bo._loss_direct(predicted, ground_truth)
    assert loss == pytest.approx(0.0)


def test_loss_direct_known_error():
    bo = BO_ecg(bo_purkinje_tree=None)

    dtype = [("I", float), ("II", float)]
    ground_truth = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)
    predicted = np.array([(2.0, 3.0), (4.0, 5.0)], dtype=dtype)  # Error 1 en cada punto

    # Error por lead: (1^2 + 1^2)/2 = 1.0 → total = 2.0
    loss = bo._loss_direct(predicted, ground_truth)
    assert loss == pytest.approx(2.0)


def test_loss_direct_missing_lead():
    bo = BO_ecg(bo_purkinje_tree=None)

    dtype_gt = [("I", float), ("II", float)]
    ground_truth = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype_gt)

    # Solo incluye el lead "I", el "II" está ausente
    dtype_pred = [("I", float)]
    predicted_missing = np.array([(1.0,), (3.0,)], dtype=dtype_pred)

    loss = bo._loss_direct(predicted_missing, ground_truth)

    # Se computa lead "I", y el "II" lanza excepción → se añade inf
    assert loss == pytest.approx(np.inf)

def test_extract_overlapping_section_valid_delay():
    bo = BO_ecg(bo_purkinje_tree=None)

    ground_truth = np.array([1, 2, 3, 4, 5])
    predicted = np.array([10, 20, 30, 40, 50])

    # Desfase de +2 → predicted se atrasa, ground_truth usa [2:5], predicted [0:3]
    delay = 2

    gt_seg, pred_seg = bo._extract_overlapping_section(ground_truth, predicted, delay)

    np.testing.assert_array_equal(gt_seg, np.array([3, 4, 5]))
    np.testing.assert_array_equal(pred_seg, np.array([10, 20, 30]))

def test_calculate_loss_direct():
    bo = BO_ecg(bo_purkinje_tree=None)

    dtype = [("I", float), ("II", float)]
    gt = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)
    pred = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dtype)

    loss = bo.calculate_loss(predicted=pred, cross_correlation=False, ecg_pat=gt)
    assert np.isclose(loss, 0.0)

from unittest.mock import MagicMock

def test_mse_jaxbo_creates_objective_function():
    mock_tree = MagicMock()
    bo = BO_ecg(bo_purkinje_tree=mock_tree)

    var_params = [
        OptimParam("cv", np.array([2.0]), np.array([4.0]), PriorType.UNIFORM)
    ]

    # Simulación falsa de ECG + pérdida
    gt = np.zeros(2, dtype=[("I", float)])
    bo.calculate_loss = MagicMock(return_value=42.0)
    mock_tree.run_ECG.return_value = (gt, None, None)

    f, p_x, bounds = bo.mse_jaxbo(gt, var_params)

    assert callable(f)
    assert "lb" in bounds and "ub" in bounds
    assert bo.dim == 1

    loss = f(np.array([3.0]))
    assert np.isclose(loss, 42.0)
    mock_tree.run_ECG.assert_called_once()

def test_set_initial_training_data_no_noise():
    bo = BO_ecg(bo_purkinje_tree=None)
    bo.f = MagicMock(side_effect=lambda x: np.sum(x**2))  # f(x) = ∑x²

    # Simular configuración previa
    bo.lb_params = np.array([0.0, 0.0])
    bo.ub_params = np.array([1.0, 1.0])
    bo.dim = 2

    X, y = bo.set_initial_training_data(N=5, noise=0.0)

    assert X.shape == (5, 2)
    assert y.shape == (5,)
    assert (y >= 0).all()
    assert bo.f.call_count == 5

def test_set_initial_training_data_with_noise():
    bo = BO_ecg(bo_purkinje_tree=None)

    bo.f = MagicMock(side_effect=lambda x: 10.0 + float(x[0]))

    bo.lb_params = np.array([0.0])
    bo.ub_params = np.array([1.0])
    bo.dim = 1

    X, y = bo.set_initial_training_data(N=10, noise=0.2)

    assert y.shape == (10,)
    assert not np.allclose(y, y[0])


def test_bo_loop_one_iteration():
    bo = BO_ecg(bo_purkinje_tree=None)

    # Simular funciones y atributos necesarios
    bo.f = MagicMock(side_effect=lambda x: float(np.sum(x**2)))
    bo.lb_params = np.array([0.0, 0.0])
    bo.ub_params = np.array([1.0, 1.0])
    bo.bounds = {"lb": bo.lb_params, "ub": bo.ub_params}
    bo.dim = 2
    bo.noise = 0.0

    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([0.05, 0.25])
    X_star = np.array([[0.5, 0.5]])

    # Configuración mínima de opciones
    options = {
        "nIter": 1,
        "criterion": "EI",
        "kernel": "RBF",
        "lr": 0.01,
        "epochs": 10,
        "kappa": 1.0,
        "input_dim": 2,
    }

    # Patch del modelo GP con predicciones y punto siguiente
    mock_gp = MagicMock()
    mock_gp.train.return_value = {}
    mock_gp.predict.return_value = (np.array([0.1, 0.2]), np.array([0.01, 0.01]))
    mock_gp.compute_next_point_lbfgs.return_value = (
        np.array([[0.6, 0.6]]), None, None
    )
    mock_gp.options = options
    bo_loop_gp_patch = patch("bo_ecg.GP", return_value=mock_gp)

    with bo_loop_gp_patch:
        X_out, y_out, _ = bo.bo_loop(X.copy(), y.copy(), X_star, None, options)

    assert X_out.shape[0] == 3
    assert y_out.shape[0] == 3
    assert bo.f.call_count == 1
