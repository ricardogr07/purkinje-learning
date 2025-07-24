import os
import time
import logging
import pickle
import shutil

import numpy as onp
import pandas as pd
import jax
import jax.numpy as np
from jaxbo.utils import normalize
from jax import random, lax
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Dict

import seaborn as sns
import matplotlib.pyplot as plt

from bo_purkinje_tree import BO_PurkinjeTree, BO_PurkinjeTreeConfig
from bo_ecg import BO_ecg, OptimParam
from jaxbo.models import GP
from bo_utils.enums import CriterionBO, TrainingDataSource, OptimizationMode, DeviceType, BOECGParameter, Prior
from myocardial_mesh import MyocardialMesh

# Optional: configure JAX
jax.config.update("jax_enable_x64", True)

# Logging setup
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Seed numpy for reproducibility
onp.random.seed(1234)


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

def load_demo_geometry(patient_number: str, device: str = "cpu"):
    if patient_number != "demo":
        raise ValueError("Only 'demo' patient is supported.")

    patient_data_path = "data/crtdemo"

    # These are node indices (of the LV and RV endocardial meshes) that determine the direction of the
    # initial branch of the Purkinje Tree
    # Here, 388 and 412 are nodes of the LV endocardial mesh and
    #       198 and 186 are nodes of the RV endocardial mesh
    meshes_list_pat = [388, 412, 198, 186]

    myocardial_mesh = MyocardialMesh(
        myo_mesh=f"{patient_data_path}/crtdemo_mesh_oriented.vtk",
        electrodes_position=f"{patient_data_path}/electrode_pos.pkl",
        fibers=f"{patient_data_path}/crtdemo_f0_oriented.vtk",
        device=device,
    )

    return meshes_list_pat, myocardial_mesh, patient_data_path


def get_reference_tree_config() -> BO_PurkinjeTreeConfig:
    """
    Returns the reference tree configuration used to simulate the ground truth.
    """
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


def initialize_reference_model(
    patient_prefix: str,
    meshes_list: list,
    config: BO_PurkinjeTreeConfig,
    myocardium: MyocardialMesh
) -> tuple[BO_PurkinjeTree, BO_ecg]:
    """
    Initializes the BO_PurkinjeTree and BO_ecg model using the provided config and geometry.

    Parameters
    ----------
    patient_prefix : str
        Path prefix for mesh files (e.g., "data/crtdemo/crtdemo").
    meshes_list : list
        Node indices for LV and RV initial direction.
    config : BO_PurkinjeTreeConfig
        Configuration for tree growth.
    myocardium : MyocardialMesh
        The myocardial domain used for activation and ECG computation.

    Returns
    -------
    tuple of (BO_PurkinjeTree, BO_ecg)
    """
    logger.info("Initializing reference Purkinje tree and BO model...")

    purkinje_tree = BO_PurkinjeTree(
        patient=patient_prefix,
        meshes_list=meshes_list,
        config=config,
        myocardium=myocardium
    )

    bo_model = BO_ecg(bo_purkinje_tree=purkinje_tree)

    return purkinje_tree, bo_model


def load_ground_truth_parameters(X_path: str, y_path: str) -> dict:
    """
    Loads the ground-truth parameters from precomputed BO results.

    Parameters
    ----------
    X_path : str
        Path to the .npy file containing sampled input parameters.
    y_path : str
        Path to the .npy file containing MSE losses.

    Returns
    -------
    dict
        Dictionary with simulation-ready ground-truth parameters.
    """
    logger.info("Loading results for ground truth...")
    
    X_read = onp.load(X_path)
    y_read = onp.load(y_path)

    X_min = X_read[onp.argmin(y_read)]
    y_min = onp.min(y_read)

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


def simulate_ecg(
    model: BO_ecg,
    parameters: dict,
    side: str = "both"
) -> tuple[onp.ndarray, object, object]:
    """
    Simulates the ECG using the provided BO_ecg model and parameter dictionary.

    Parameters
    ----------
    model : BO_ecg
        The model that wraps a BO_PurkinjeTree instance.
    parameters : dict
        Dictionary of parameter values to apply.
    side : str
        Which side(s) to apply the parameters to (default: 'both').

    Returns
    -------
    tuple of (ecg, LVtree, RVtree)
    """
    logger.info("Simulating reference ECG with true parameters...")

    ecg, LVtree, RVtree = model.bo_purkinje_tree.run_ECG(
        n_sim=0,
        modify=True,
        side=side,
        **parameters
    )

    return ecg, LVtree, RVtree


def save_reference_outputs(
    output_dir: str,
    ecg: onp.ndarray,
    myocardium: MyocardialMesh,
    LVtree,
    RVtree
) -> None:
    """
    Saves the reference ECG, endocardial activation, and Purkinje trees to disk.

    Parameters
    ----------
    output_dir : str
        Directory where files will be saved.
    ecg : np.ndarray
        The simulated reference ECG array.
    myocardium : MyocardialMesh
        The myocardium object to export as VTU.
    LVtree : object
        Left ventricular PurkinjeTree instance to save.
    RVtree : object
        Right ventricular PurkinjeTree instance to save.
    """
    logger.info("Saving reference ECG and geometries...")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "True_ecg"), "wb") as f:
        pickle.dump(ecg, f)

    myocardium.save_pv(os.path.join(output_dir, "True_endo.vtu"))
    LVtree.save(os.path.join(output_dir, "True_LVtree.vtu"))
    RVtree.save(os.path.join(output_dir, "True_RVtree.vtu"))


def plot_ecg(ecg_array: onp.ndarray, title: str = "Reference ECG") -> None:
    """
    Plots the 12-lead ECG array using a fixed grid layout.

    Parameters
    ----------
    ecg_array : np.ndarray
        Structured ECG array with named leads.
    title : str
        Title shown on top of the plot.
    """
    logger.info("Plotting reference ECG...")

    fig, axs = plt.subplots(3, 4, figsize=(10, 13), dpi=120, sharex=True, sharey=True)

    for ax, lead in zip(axs.ravel(), ecg_array.dtype.names):
        ax.plot(ecg_array[lead])
        ax.grid()
        ax.set_title(lead)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def var_ecg_parameters_list(var_parameters_names: List[BOECGParameter], dim=2, prior=Prior.UNIFORM) -> List[OptimParam]:

    var_parameters = []

    if BOECGParameter.INIT_LENGTH in var_parameters_names:
        lb = 30.0 * onp.ones(dim)
        ub = 100.0 * onp.ones(dim)
        var_parameters.append(OptimParam(BOECGParameter.INIT_LENGTH.value, lb, ub, prior.value))

    if BOECGParameter.LENGTH in var_parameters_names:
        lb = 4.0 * onp.ones(1)
        ub = 12.0 * onp.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.LENGTH.value, lb, ub, prior.value))

    if BOECGParameter.W in var_parameters_names:
        lb = 0.05 * onp.ones(1)
        ub = 0.25 * onp.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.W.value, lb, ub, prior.value))

    if BOECGParameter.L_SEGMENT in var_parameters_names:
        lb = 1.0 * onp.ones(dim)
        ub = 15.0 * onp.ones(dim)
        var_parameters.append(OptimParam(BOECGParameter.L_SEGMENT.value, lb, ub, prior.value))

    if BOECGParameter.FASCICLES_LENGTH in var_parameters_names:
        lb = 2.0 * onp.ones(2 * dim)
        ub = 50.0 * onp.ones(2 * dim)
        var_parameters.append(OptimParam(BOECGParameter.FASCICLES_LENGTH.value, lb, ub, prior.value))

    if BOECGParameter.FASCICLES_ANGLES in var_parameters_names:
        lb = -1.0 / 4.0 * onp.pi * np.ones(2 * dim)
        ub =  3.0 / 4.0 * onp.pi * np.ones(2 * dim)
        var_parameters.append(OptimParam(BOECGParameter.FASCICLES_ANGLES.value, lb, ub, prior.value))

    if BOECGParameter.BRANCH_ANGLE in var_parameters_names:
        lb = 5.0 * onp.pi / 180.0 * np.ones(1)
        ub = 45.0 * onp.pi / 180.0 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.BRANCH_ANGLE.value, lb, ub, prior.value))

    if BOECGParameter.ROOT_TIME in var_parameters_names:
        lb = -75.0 * np.ones(1)
        ub = 50.0 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.ROOT_TIME.value, lb, ub, prior.value))

    if BOECGParameter.CV in var_parameters_names:
        lb = 2.0 * np.ones(1)
        ub = 4.0 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.CV.value, lb, ub, prior.value))

    return var_parameters


def generate_file_prefix(patient_number: int, N: int, var_parameters: list[OptimParam]) -> str:
    var_names = "_".join([p.parameter for p in var_parameters])
    return os.path.join("output", f"patient{patient_number}", f"data_{N}_{var_names}")


def initial_values(
    var_parameters: List[OptimParam],
    N_it: int = 20
) -> BO_PurkinjeTreeConfig:
    
    param_dict = {p.parameter: float(p.lower.mean()) for p in var_parameters}

    return BO_PurkinjeTreeConfig(
        init_length      = param_dict.get("init_length", 30.0),
        length           = param_dict.get("length", 8.0),
        w                = param_dict.get("w", 0.1),
        l_segment        = param_dict.get("l_segment", 1.0),
        fascicles_length = param_dict.get("fascicles_length", [20.0, 20.0]),
        fascicles_angles = param_dict.get("fascicles_angles", [1.0, 1.0]),
        branch_angle     = param_dict.get("branch_angle", 0.15),
        N_it             = N_it
    )


def plot_pairplot(X: np.ndarray, y: np.ndarray, var_ecg_parameters: List[OptimParam], enabled: bool = False):
    """
    Plots a pairplot of the Bayesian Optimization samples if enabled.

    Parameters
    ----------
    X : np.ndarray
        Input parameters from BO (samples × dimensions).
    y : np.ndarray
        Corresponding objective function values.
    var_ecg_parameters : List[OptimParam]
        List of optimized parameters, including names.
    enabled : bool
        Whether to generate the plot.
    """
    if not enabled:
        return

    param_names = [p.parameter for p in var_ecg_parameters]
    df_columns = []

    if "init_length" in param_names:
        df_columns += ["In. Length L", "In. Length R"]
    if "length" in param_names:
        df_columns += ["Length"]
    if "w" in param_names:
        df_columns += ["w"]
    if "fascicles_length" in param_names:
        df_columns += ["Fas. Length L1", "Fas. Length L2", "Fas. Length R1", "Fas. Length R2"]
    if "fascicles_angles" in param_names:
        df_columns += ["Fas. Angle L1", "Fas. Angle L2", "Fas. Angle R1", "Fas. Angle R2"]
    if "branch_angle" in param_names:
        df_columns += ["Branch Angle"]
    if "root_time" in param_names:
        df_columns += ["Root time"]
    if "cv" in param_names:
        df_columns += ["CV"]

    df = pd.DataFrame(X, columns=df_columns)
    df["y"] = y

    num_bins = 5
    bin_labels = [f"Bin {i}" for i in range(1, num_bins + 1)]
    df["hue_bins"] = pd.cut(df["y"], bins=num_bins, labels=bin_labels)

    sns.pairplot(df, hue="hue_bins")
    plt.show()


def find_std_ybest_ecgs(
    X: np.ndarray,
    y: np.ndarray,
    qrs_in: int,
    qrs_fin: int,
    var_ecg_parameters: List[OptimParam],
    bo_method: BO_ecg,
    ecg_patient: dict[str, np.ndarray],
) -> List[float]:
    """
    Compute MSE values between best predicted ECG and each patient's ECG instance.

    Parameters
    ----------
    X : np.ndarray
        Sampled input parameters (N × dim).
    y : np.ndarray
        Corresponding MSE values.
    qrs_in : int
        Start index of the QRS complex window.
    qrs_fin : int
        End index of the QRS complex window.
    var_ecg_parameters : List[OptimParam]
        Variable ECG parameters optimized during BO.
    bo_method : BO_ecg
        Instance of the Bayesian Optimization ECG class.
    ecg_patient : dict[str, np.ndarray]
        Dictionary of ECG leads: each value is (T × M) where M is number of instances.

    Returns
    -------
    List[float]
        MSE values comparing best ECG to each patient instance.
    """

    bo_method.logger.info("Finding MSE between best ECG prediction and patient ECG instances...")

    # Get the input X with minimum loss
    X_min = X[onp.argmin(y)]

    # Run ECG simulation using best parameters
    ecg_min, _, _ = bo_method.update_purkinje_tree(X_min[None, :], y.min(), var_ecg_parameters)

    bo_method.logger.info(f"Best input parameters: {X_min}")
    bo_method.logger.info(f"Best MSE value: {y.min()}")

    # Compute MSE against each ECG instance in the patient dataset
    mse_values_best = []
    lead_names = list(ecg_patient.keys())
    num_instances = ecg_patient[lead_names[0]].shape[1]

    for i in range(num_instances):
        ecg_instance = []
        for lead in lead_names:
            ecg_instance.append(ecg_patient[lead][:, i] / 1e3)

        ecg_patient_array = onp.rec.fromarrays(ecg_instance, names=lead_names)
        ecg_patient_window = ecg_patient_array[qrs_in:qrs_fin]

        mse = bo_method.calculate_loss(predicted=ecg_min, ecg_pat=ecg_patient_window)
        mse_values_best.append(mse)

    return mse_values_best


def train_gp_model(
    X: np.ndarray,
    y: np.ndarray,
    options: dict,
    bo_method: BO_ecg,
    X_star_uniform: np.ndarray,
    gp_state: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, Any]:
    """
    Train a GP model using valid points or reuse a previous GP model to predict over test points.

    Parameters
    ----------
    X : np.ndarray
        Training input data.
    y : np.ndarray
        Corresponding MSE values.
    options : dict
        GP model options (passed to GP constructor).
    bo_method : BO_ecg
        BO_ecg instance with parameter bounds and logging.
    X_star_uniform : np.ndarray
        Test points to evaluate GP predictions.
    gp_state : Optional[Any]
        Previously trained GP state to reuse (default: None).

    Returns
    -------
    ys : np.ndarray
        Predicted mean values at test points.
    sigmas : np.ndarray
        Predicted standard deviations at test points.
    y_gp_best : float
        GP prediction at the best observed input.
    sigma_gp_best : float
        Std of GP prediction at the best observed input.
    gp_state : Any
        Trained GP model and normalization state for reuse.
    """

    if gp_state is None:
        bo_method.logger.info("Training GP model with valid points...")

        valid = y != bo_method.y_trees_non_valid
        X_valid = X[valid]
        y_valid = y[valid]

        bo_method.logger.info(f"{len(X_valid)} valid training points found.")

        rng_key = random.PRNGKey(0)
        gp_model = GP(options)

        # Normalize training data
        norm_batch, norm_const = normalize(X_valid, y_valid, bo_method.bounds)

        # Train GP model
        t_ini = time.time()
        opt_params = gp_model.train(norm_batch, rng_key, num_restarts=5)
        t_end = time.time()

        bo_method.logger.info(f"GP training completed in {t_end - t_ini:.2f} seconds")

        kwargs = {
            "params": opt_params,
            "batch": norm_batch,
            "norm_const": norm_const,
            "bounds": bo_method.bounds,
        }

        gp_state = [gp_model, [norm_batch, norm_const], kwargs]
    else:
        bo_method.logger.info("Reusing previous GP model (no retraining).")
        gp_model = gp_state[0]
        norm_batch, norm_const = gp_state[1]
        kwargs = gp_state[2]

    # Predict mean and std over test set
    t_ini = time.time()

    n_col = 100
    assert (len(X_star_uniform) / n_col).is_integer(), "Please modify n_col to match test set size."

    reshaped_X = [X_star_uniform[i:i + n_col] for i in range(0, len(X_star_uniform), n_col)]
    mean_batch, std_batch = lax.map(lambda x: gp_model.predict(x, **kwargs), np.array(reshaped_X))

    mean_batch = mean_batch.flatten()
    std_batch = std_batch.flatten()

    t_end = time.time()
    bo_method.logger.info(f"GP predictions computed in {t_end - t_ini:.2f} seconds")

    # Un-normalize predictions
    ys = mean_batch * norm_const["sigma_y"] + norm_const["mu_y"]
    sigmas = std_batch * norm_const["sigma_y"]

    # Prediction at best X
    X_min = X[np.argmin(y)]
    mean_min, std_min = gp_model.predict(X_min[None, :], **kwargs)
    y_gp_best = mean_min[0] * norm_const["sigma_y"] + norm_const["mu_y"]
    sigma_gp_best = std_min[0] * norm_const["sigma_y"]

    return ys, sigmas, y_gp_best, sigma_gp_best, gp_state


def rejection_sampling(
    ys: np.ndarray,
    sigmas: np.ndarray,
    y_gp_best: float,
    sigma_gp_best: float,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform rejection sampling to select promising candidates based on GP predictions.

    Parameters
    ----------
    ys : np.ndarray
        GP predicted means at candidate points.
    sigmas : np.ndarray
        GP predicted standard deviations at candidate points.
    y_gp_best : float
        Best (lowest) predicted value from the GP model.
    sigma_gp_best : float
        Standard deviation of the best GP prediction.
    logger : logging.Logger
        Logger instance for debug messages.

    Returns
    -------
    accepted_samples : np.ndarray
        Boolean mask of accepted samples.
    comparison : np.ndarray
        Uniform random threshold values for acceptance.
    likelihoods : np.ndarray
        Likelihood values based on GP predictive uncertainty.
    """
    logger.info("Starting rejection sampling...")

    max_likelihood = norm.pdf(
        x=0.0,
        loc=0.0,
        scale=np.sqrt(sigma_gp_best**2 + np.min(sigmas)**2)
    )
    logger.debug(f"Max likelihood value for scaling: {max_likelihood:.4e}")

    key = random.PRNGKey(0)
    comparison = random.uniform(key, shape=(ys.shape[0],)) * max_likelihood

    likelihoods = norm.pdf(
        x=0.0,
        loc=ys - y_gp_best,
        scale=np.sqrt(sigmas**2 + sigma_gp_best**2)
    )

    accepted_samples = likelihoods > comparison
    n_accepted = int(np.sum(accepted_samples))

    logger.info(f"{n_accepted} samples accepted out of {len(ys)}")

    return accepted_samples, comparison, likelihoods


def check_accepted_samples(
    N_samples: int,
    X_star_uniform: np.ndarray,
    y_gp_best: float,
    accepted_samples: np.ndarray,
    comparison: np.ndarray,
    likelihoods: np.ndarray,
    var_parameters: List[OptimParam],
    bo_class: Any,
    tol: float,
    observed_samples: List[Any],
    confirmed_samples: Dict[str, Any],
    folder_trees: str,
    patient_number: int,
    N: int,
    nIter: int,
    options: Dict[str, Any],
    start_time: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    Check accepted samples by evaluating the predicted ECG and loss, saving those under a threshold.

    Returns
    -------
    Tuple[str, Dict]
        "ok" and confirmed_samples if enough samples accepted,
        otherwise "retrain gp" and info_final with new samples.
    """
    X_accepted = X_star_uniform[accepted_samples]
    comparison_accepted = comparison[accepted_samples]
    likelihoods_accepted = likelihoods[accepted_samples]
    sorted_indices = np.argsort(-likelihoods_accepted)

    X_accepted = X_accepted[sorted_indices]
    comparison_accepted = comparison_accepted[sorted_indices]

    X_true_new, Y_true_new = [], []
    state = []
    observed_samples_new = []

    if not confirmed_samples:
        confirmed_samples["samples_final"] = []
        confirmed_samples["ecg_final"] = []
        confirmed_samples["Tree_final"] = []
        confirmed_samples["loss_final"] = []

    logger.log(f"tolerance: {tol}")
    X_ind = 1
    n_candidates = len(X_accepted)

    for x_accepted, comp in zip(X_accepted, comparison_accepted):
        logger.log(f"Checking point {X_ind}/{n_candidates}")

        if any(np.array_equal(obs[0], x_accepted) for obs in observed_samples):
            logger.log("Point already observed")
            X_ind += 1
            continue

        try:
            ecg_i, LVtree_i, RVtree_i = bo_class.update_purkinje_tree(
                np.array([x_accepted]), np.array([1.0]), var_parameters
            )
            endo_i = bo_class.endo
        except Exception:
            logger.log("The fascicle goes out of the domain")
            X_true_new.append(x_accepted)
            Y_true_new.append(bo_class.y_trees_non_valid)
            state.append("rejected")
            observed_samples_new.append([x_accepted, "non_valid"])
            X_ind += 1
            continue

        loss_i, ind_loss_i = bo_class.calculate_loss(
            predicted=ecg_i, cross_correlation=True, return_ind=True
        )
        observed_samples_new.append([
            x_accepted, [ecg_i, endo_i, LVtree_i, RVtree_i, loss_i, ind_loss_i]
        ])

        logger.log(f"Loss: {loss_i}")
        X_true_new.append(x_accepted)
        Y_true_new.append(loss_i)

        err_value = loss_i - y_gp_best
        logger.log(f"y_true - y_gp_best: {err_value}")

        if err_value < tol:
            state.append("accepted")
            confirmed_samples["samples_final"].append(x_accepted)
            confirmed_samples["ecg_final"].append([ecg_i, ind_loss_i])
            confirmed_samples["Tree_final"].append([LVtree_i, RVtree_i])
            confirmed_samples["loss_final"].append(loss_i)

            n_conf = len(confirmed_samples["samples_final"])
            logger.log(f"Sample accepted! (n°{n_conf-1}, {n_conf}/{N_samples})")

            tree_ind_test = n_conf - 1
            LVtree_i.save(
                f"./output/patient{patient_number}{folder_trees}/LVtree_N{N}_nIter{nIter}_criterion{options['criterion']}_{tree_ind_test}.vtu"
            )
            RVtree_i.save(
                f"./output/patient{patient_number}{folder_trees}/RVtree_N{N}_nIter{nIter}_criterion{options['criterion']}_{tree_ind_test}.vtu"
            )
            endo_i.save_pv(
                f"./output/patient{patient_number}{folder_trees}/propeiko_N{N}_nIter{nIter}_criterion{options['criterion']}_{tree_ind_test}.vtu"
            )
        else:
            logger.log("Sample rejected")
            state.append("rejected")

        if len(confirmed_samples["samples_final"]) >= N_samples:
            return "ok", confirmed_samples

        if len(state) >= 50 and state[-50:] == ["rejected"] * 50:
            logger.log(f"Elapsed time: {time.time() - start_time}")
            logger.log(f"No sample accepted in the last 50 iterations ({len(confirmed_samples['samples_final'])} accepted)")
            return "retrain gp", {
                "X_true_new": X_true_new,
                "Y_true_new": Y_true_new,
                "Observed_samples": observed_samples + observed_samples_new,
                "Confirmed_samples": confirmed_samples,
            }

        X_ind += 1

    logger.log(f"Elapsed time: {time.time() - start_time}")
    logger.log(f"All samples were observed, but only {len(confirmed_samples['samples_final'])} accepted.")
    return "retrain gp", {
        "X_true_new": X_true_new,
        "Y_true_new": Y_true_new,
        "Observed_samples": observed_samples + observed_samples_new,
        "Confirmed_samples": confirmed_samples,
    }


def run_rejection_sampling_loop(
    bo_method,
    X: np.ndarray,
    y: np.ndarray,
    var_parameters: List[OptimParam],
    patient_number: int,
    N: int,
    nIter: int,
    criterion_bo: str,
    list_variable_params: str,
    start_time: float,
    N_samples: int = 30,
    tol: float = 100.0,
    nn: int = 5_000_000,
    max_rejection_loops: int = 50
) -> Tuple[List[np.ndarray], List[Any], List[Any], List[float]]:
    """
    Run the rejection sampling loop to find and accept N_samples based on a GP surrogate model.

    Parameters
    ----------
    bo_method : BO_ecg
        Bayesian optimization wrapper class.
    X : np.ndarray
        Input parameter values used for training the GP model.
    y : np.ndarray
        Corresponding loss values.
    var_parameters : List[OptimParam]
        Parameters and bounds used in the optimization.
    patient_number : int
        Identifier for the patient folder.
    N : int
        Number of initial training points.
    nIter : int
        Number of optimization iterations.
    criterion_bo : str
        Acquisition function used.
    list_variable_params : str
        String with parameter names used in the filename.
    N_samples : int
        Number of final samples to accept.
    tol : float
        Loss tolerance to accept samples.
    nn : int
        Number of uniform samples in each rejection loop.
    max_rejection_loops : int
        Maximum number of rejection loops before raising an exception.

    Returns
    -------
    Tuple[List[np.ndarray], List[Any], List[Any], List[float]]
        Final samples, ECGs, trees, and their losses.
    """

    options = {
        'kernel': 'Matern12',
        'criterion': criterion_bo,
        'input_prior': bo_method.prior,
        'kappa': 2.0,
        'nIter': nIter,
    }

    gp_rejection = None
    rejection_n = 1
    obs_samples = []
    conf_samples = {}

    folder_trees = f"/Trees_N{N}_nIter{nIter}_criterion{criterion_bo}_variableparams_{list_variable_params}"
    folder_path = f"./output/patient{patient_number}{folder_trees}"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    while True:
        logger.log(f"\nRejection loop {rejection_n}")
        y_best = np.min(y)

        key = random.PRNGKey(rejection_n - 1)
        X_star_uni = bo_method.lb_params + (bo_method.ub_params - bo_method.lb_params) * \
            random.uniform(key, shape=(nn, bo_method.dim))

        ys, sigmas, y_gp_best, sigma_gp_best, gp_rejection = train_gp_model(
            X, y, options, bo_class=bo_method, X_star_uniform=X_star_uni, gp_state=None
        )

        accepted_samples, comparison, likelihoods = rejection_sampling(
            ys, sigmas, y_gp_best, sigma_gp_best
        )

        state_final, info_final = check_accepted_samples(
            N_samples=N_samples,
            X_star_uniform=X_star_uni,
            y_gp_best=y_gp_best,
            accepted_samples=accepted_samples,
            comparison=comparison,
            likelihoods=likelihoods,
            var_parameters=var_parameters,
            bo_class=bo_method,
            tol=tol,
            observed_samples=obs_samples,
            confirmed_samples=conf_samples,
            folder_trees=folder_trees,
            patient_number=patient_number,
            N=N,
            nIter=nIter,
            options=options,
            start_time=start_time,
        )


        if state_final == "ok":
            samples_final = info_final["samples_final"]
            ecg_final = info_final["ecg_final"]
            Tree_final = info_final["Tree_final"]
            loss_final = info_final["loss_final"]

            onp.save(f"./output/patient{patient_number}/rejection_X_N_{N}_nIter_{nIter}_criterion_{criterion_bo}_nn_{nn}_tol_{tol}_rejection_n_{rejection_n}", X)
            onp.save(f"./output/patient{patient_number}/rejection_y_N_{N}_nIter_{nIter}_criterion_{criterion_bo}_nn_{nn}_tol_{tol}_rejection_n_{rejection_n}", y)
            return samples_final, ecg_final, Tree_final, loss_final

        elif state_final == "retrain gp":
            obs_samples = info_final["Observed_samples"]
            conf_samples = info_final["Confirmed_samples"]

            X_new = info_final["X_true_new"]
            y_new = np.asarray(info_final["Y_true_new"])

            X = np.concatenate([X, np.array(X_new)], axis=0)
            y = np.concatenate([y, y_new], axis=0)

            assert len(X_new) == len(y_new), "Mismatch in added samples"
            logger.log(f"Retrain the GP model with {len(X_new)} new points")

        if rejection_n == max_rejection_loops:
            onp.save(f"./output/patient{patient_number}/rejection_X_N_{N}_nIter_{nIter}_criterion_{criterion_bo}_nn_{nn}_tol_{tol}_rejection_n_{rejection_n}", X)
            onp.save(f"./output/patient{patient_number}/rejection_y_N_{N}_nIter_{nIter}_criterion_{criterion_bo}_nn_{nn}_tol_{tol}_rejection_n_{rejection_n}", y)
            logger.error(f"Reached maximum rejection loops ({max_rejection_loops}) without finding {N_samples} accepted samples.")
            raise RuntimeError(f"Could not find {N_samples} accepted samples after {max_rejection_loops} loops.")

        rejection_n += 1


def main(config: PipelineConfig):

    start_time = time.time()

    logger.info("#" * 40)
    logger.info(f"Running analysis of patient {config.patient_number}")
    logger.info(
        f"Pipeline parameters: N={config.N}, nIter={config.nIter}, "
        f"obtain_training_data={config.obtain_training_data}, "
        f"criterion_bo={config.criterion_bo}, optimization_points={config.optimization_points}"
    )
    logger.info("#" * 40)

    meshes_list_pat, myo, patient_data_path = load_demo_geometry(
        config.patient_number, device=config.device
    )

    true_parameters_config = get_reference_tree_config()

    patient_prefix = os.path.join(patient_data_path, "crtdemo")

    _, bo_method_true = initialize_reference_model(
        patient_prefix,
        meshes_list_pat,
        true_parameters_config,
        myo
    )

    output_dir = f"./output/patient{config.patient_number}"
    os.makedirs(output_dir, exist_ok=True)

    # This string is static here — if it becomes parametric, move to config
    var_parameters_str = "init_length_fascicles_length_fascicles_angles_root_time_cv"
    X_path = f"output/patient1/data_X_N_250_nIter_300_criterionEI_{var_parameters_str}.npy"
    y_path = f"output/patient1/data_y_N_250_nIter_300_criterionEI_{var_parameters_str}.npy"

    var_params_true = load_ground_truth_parameters(X_path, y_path)

    ecg_true, LVtree_true, RVtree_true = simulate_ecg(
        model=bo_method_true,
        parameters=var_params_true,
        side="both"
    )

    save_reference_outputs(
        output_dir=output_dir,
        ecg=ecg_true,
        myocardium=myo,
        LVtree=LVtree_true,
        RVtree=RVtree_true
    )

    qrs_in, qrs_fin = 0, len(ecg_true)
    ecg_pat_array = ecg_true[qrs_in:qrs_fin]

    if config.plot:
        plot_ecg(ecg_array=ecg_pat_array)

    logger.info("Setting up Bayesian Optimization parameters...")

    var_ecg_parameters = var_ecg_parameters_list(config.var_ecg_parameters, dim=2, prior=config.prior)
    logger.info(f"Variable ECG parameters for BO: {[p.parameter for p in var_ecg_parameters]}")

    config_bo = initial_values(var_ecg_parameters)
    logger.info(f"Initial values for BO_PurkinjeTreeConfig: {config_bo}")

    logger.info("Initializing BO_PurkinjeTree for optimization...")
    Tree_bo = BO_PurkinjeTree(
        patient=patient_prefix, # e.g., "data/crtdemo/crtdemo"
        meshes_list=meshes_list_pat,
        config=config_bo,
        myocardium=myo
    )

    logger.info("Wrapping BO_PurkinjeTree with BO_ecg...")
    bo_method = BO_ecg(bo_purkinje_tree=Tree_bo, N=config.N)

    logger.info("Starting Bayesian Optimization with mse_jaxbo...")

    f, p_x_params, _ = bo_method.mse_jaxbo(
        ground_truth=ecg_pat_array,
        variable_parameters=var_ecg_parameters
    )

    logger.debug(f"Objective function: {f}, Expected input shape: {p_x_params.shape if hasattr(p_x_params, 'shape') else 'N/A'}")
    logger.info("Bayesian Optimization setup complete.")

    x = np.array([
        74.12523435, 62.46094801,           # init_length (2)
        24.19023292, 40.65849346,           # fascicles_length[0]
        32.69169548, 49.33628498,           # fascicles_length[1]
        1.08097521, -0.73921825,           # fascicles_angles[0]
        0.74647183,  1.68333156,           # fascicles_angles[1]
        12.50440764,                        # root_time (1)
        2.5114008                          # cv (1)
    ])

    param_dict = bo_method.set_dictionary_variables(bo_method.variable_parameters, x)
    logger.info("Generated parameter dictionary for run_ECG:")
    for k, v in param_dict.items():
        logger.info(f"  {k}: {v}")

    if config.obtain_training_data == TrainingDataSource.COMPUTE:
        logger.info("Computing training points...")
        t_ini_train = time.time()

        bo_method.y_trees_non_valid = 10.0
        noise = 0.0
        X, y = bo_method.set_initial_training_data(config.N, noise)

        file_prefix = generate_file_prefix(config.patient_number, config.N, bo_method.variable_parameters)
        os.makedirs(os.path.dirname(file_prefix), exist_ok=True)

        onp.save(file_prefix.replace("data_", "data_X_"), X)
        onp.save(file_prefix.replace("data_", "data_y_"), y)
        onp.save(file_prefix.replace("data_", "data_noise_"), noise)

        elapsed = time.time() - t_ini_train
        logger.info(f"Training data generated in {elapsed:.2f} seconds.")

    elif config.obtain_training_data == TrainingDataSource.LOAD:
        logger.info("Loading training points...")

        file_prefix = generate_file_prefix(config.patient_number, config.N, bo_method.variable_parameters)

        X = np.load(file_prefix.replace("data_", "data_X_") + ".npy")
        y = np.load(file_prefix.replace("data_", "data_y_") + ".npy")
        noise = np.load(file_prefix.replace("data_", "data_noise_") + ".npy")

        bo_method.noise = noise
        logger.info(f"Training data loaded from {file_prefix}_*.npy")
    
    # Assign max loss to non-valid trees (initially marked with y = 10.)
    valid = y != 10.0
    y_valid = y[valid]

    bo_method.y_trees_non_valid = np.max(y_valid)

    # Replace all invalid outputs with the penalty value
    mask = np.where(y == 10.0)[0]
    y = y.at[mask].set(bo_method.y_trees_non_valid)

    logger.info(f"Assigned max valid loss ({bo_method.y_trees_non_valid}) to {len(mask)} non-valid trees.")

    # Generate test points for prediction/visualization
    if bo_method.dim == 2:
        X_star, XX, YY = bo_method.set_test_data()
        logger.info("Generated 2D test grid for visualization.")
    else:
        X_star = bo_method.set_test_data()
        logger.info(f"Generated test points in {bo_method.dim}D space.")

    # Reference true global minimum (only used for visualization or evaluation)
    true_x = list(var_params_true.values())
    logger.info(f"True x values (global minimum reference): {true_x}")

    # Set optimization options
    options = {
        "kernel": "Matern12",  # Could also try 'Matern52'
        "criterion": config.criterion_bo,
        "input_prior": p_x_params,
        "kappa": 2.0,
        "nIter": config.nIter,
    }

    opt_file_prefix = (
        f"./output/patient{config.patient_number}/data_N_{config.N}_nIter_{config.nIter}_criterion{config.criterion_bo}_"
        + "_".join([p.parameter for p in var_ecg_parameters])
    )

    if config.optimization_points == OptimizationMode.RUN_OPT:
        logger.info("Running Bayesian Optimization loop ...")
        t_ini_opt = time.time()

        X, y, info_iterations = bo_method.bo_loop(X, y, X_star, true_x, options)
        t_fin_opt = time.time()
        logger.info(f"Optimization completed in {t_fin_opt - t_ini_opt:.2f} seconds.")

        # Save optimization data
        onp.save(f"{opt_file_prefix}_X.npy", X)
        onp.save(f"{opt_file_prefix}_y.npy", y)

    elif config.optimization_points == OptimizationMode.LOAD_OPT:
        logger.info("Loading Bayesian Optimization points from disk ...")
        bo_method.nIter = config.nIter

        X = np.load(f"{opt_file_prefix}_X.npy")
        y = np.load(f"{opt_file_prefix}_y.npy")

    # Plotting MSE results (optional)
    list_variable_params = "_".join([p.parameter for p in var_ecg_parameters])
    file_name = (
        f"./output/patient{config.patient_number}/BO_N{config.N}_nIter{config.nIter}_"
        f"criterion{options['criterion']}_variableparams_{list_variable_params}"
    )

    if config.plot:
        logger.info("Plotting MSE across iterations...")
        bo_method.plot_mse(X, y, config.N, file_name)
    else:
        logger.info("Skipping MSE plot (disabled in config)")

    ecg_bo, LVtree_bo, RVtree_bo = bo_method.update_purkinje_tree(X, y, var_ecg_parameters)

    if config.plot:
        # Plot the best ecg found by the BO along with the reference ecg
        bo_method.plot_ecg_match(predicted = ecg_bo, filename_match = file_name)

    # Save tree
    bo_method.bo_purkinje_tree.myocardium.save_pv(file_name+"_myo.vtu")
    LVtree_bo.save(file_name+"_LVtree.vtu")
    RVtree_bo.save(file_name+"_RVtree.vtu")

    plot_pairplot(X, y, var_ecg_parameters, enabled=config.plot)

    samples_final, ecg_final, Tree_final, loss_final = run_rejection_sampling_loop(
        X=X,
        y=y,
        bo_method=bo_method,
        var_parameters=var_ecg_parameters,
        patient_number=config.patient_number,
        N=config.N,
        nIter=config.nIter,
        criterion_bo=config.criterion_bo,
        list_variable_params=list_variable_params,
        start_time=start_time
    )

    save_final = True

    if save_final:
        pickle.dump(ecg_final, open(f"./output/patient{config.patient_number}/ecg_N{config.N}_nIter{config.nIter}_criterion{config.criterion_bo}_variableparams_{list_variable_params}","wb"))
        onp.save(f"./output/patient{config.patient_number}/X_final_N{config.N}_nIter{config.nIter}_criterion{config.criterion_bo}_variableparams_{list_variable_params}.npy",samples_final)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":

    config = PipelineConfig(
        patient_number       = "demo",
        N                    = 250,
        nIter                = 300,
        criterion_bo         = CriterionBO.EI,
        obtain_training_data = TrainingDataSource.COMPUTE,
        optimization_points  = OptimizationMode.RUN_OPT,
        device               = DeviceType.CPU,
        plot                 = False,
        prior                = Prior.UNIFORM,
        var_ecg_parameters = [
            BOECGParameter.INIT_LENGTH,
            BOECGParameter.FASCICLES_LENGTH,
            BOECGParameter.FASCICLES_ANGLES,
            BOECGParameter.ROOT_TIME,
            BOECGParameter.CV
        ]
    )


    main(config)