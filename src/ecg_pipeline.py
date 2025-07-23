import os
import time
import pickle
import shutil
import logging

import numpy as onp
import jax
import jax.numpy as np
from jax import random

from scipy.stats import norm

from bo_purkinje_tree import BO_PurkinjeTreeConfig, BO_PurkinjeTree
from myocardial_mesh import MyocardialMesh
from bo_ecg import BO_ecg

from jaxbo.models import GP
from jaxbo.utils import normalize

from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
)
jax.config.update("jax_enable_x64", True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ecg_pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def prepare_patient(patient_number="demo"):
    logger.info(f"Loading patient '{patient_number}' data")

    if patient_number != "demo":
        raise ValueError("Only 'demo' patient is supported.")

    patient = "PurkinjeECG/data/crtdemo/crtdemo"
    meshes_list_pat = [388, 412, 198, 186]

    myo = MyocardialMesh(
        myo_mesh=f"{patient}_mesh_oriented.vtk",
        electrodes_position="PurkinjeECG/data/crtdemo/electrode_pos.pkl",
        fibers=f"{patient}_f0_oriented.vtk",
        device="cpu",
    )

    output_dir = f"./output/patient{patient_number}"
    os.makedirs(output_dir, exist_ok=True)

    return patient, meshes_list_pat, myo, output_dir


def _handle_non_valid_targets(y, bo_method, invalid_value=10.0):
    """Replace invalid y values and set y_trees_non_valid in bo_method."""
    valid = y != invalid_value
    y_valid = y[valid]

    if len(y_valid) == 0:
        raise ValueError(
            "All y values are invalid. Cannot compute non-valid threshold."
        )

    bo_method.y_trees_non_valid = onp.max(y_valid)
    y[y == invalid_value] = bo_method.y_trees_non_valid
    return y


def load_initial_data(N, out_dir, var_parameters, bo_method):
    var_key = "_".join(list(var_parameters.keys()))
    X_path = os.path.join(out_dir, f"data_X_{N}_{var_key}.npy")
    y_path = os.path.join(out_dir, f"data_y_{N}_{var_key}.npy")
    noise_path = os.path.join(out_dir, f"data_noise_{N}_{var_key}.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError("Initial data files not found. Cannot load.")

    logger.info("Loading training data from disk")
    X = onp.load(X_path)
    y = onp.load(y_path)
    noise = onp.load(noise_path)
    bo_method.noise = noise

    y = _handle_non_valid_targets(y, bo_method)
    logger.info(f"Loaded: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def load_ground_truth_parameters(patient_number, expected_dim=None):
    if patient_number != "demo":
        raise ValueError(f"No ground truth defined for patient '{patient_number}'")

    suffix = "N_250_nIter_300_criterionEI_init_length_fascicles_length_fascicles_angles_root_time_cv"
    base_path = f"./output/patient{patient_number}"
    path_X = os.path.join(base_path, f"data_X_{suffix}.npy")
    path_y = os.path.join(base_path, f"data_y_{suffix}.npy")

    if not os.path.exists(path_X) or not os.path.exists(path_y):
        raise FileNotFoundError(f"Cannot load ground truth from: {path_X} / {path_y}")

    logger.info(
        f"Loading ground truth from optimized data:\n  X: {path_X}\n  y: {path_y}"
    )

    X_all = onp.load(path_X)
    y_all = onp.load(path_y)

    idx_min = onp.argmin(y_all)
    true_x = X_all[idx_min]
    logger.info(f"Loaded ground truth vector (min loss sample) with index {idx_min}")

    if expected_dim is not None and len(true_x) != expected_dim:
        logger.warning(
            f"Ground truth length ({len(true_x)}) does not match expected input dimension ({expected_dim})"
        )
    else:
        logger.info("Ground truth parameter dimension matches expected input shape")

    return true_x


def generate_initial_data(N, output_dir, var_parameters, bo_method):
    var_key = "_".join(list(var_parameters.keys()))

    logger.info("Generating initial training data")
    noise = 0.0
    bo_method.y_trees_non_valid = 10.0
    X, y = bo_method.set_initial_training_data(N, noise)

    # Save to disk
    onp.save(os.path.join(output_dir, f"data_X_{N}_{var_key}.npy"), X)
    onp.save(os.path.join(output_dir, f"data_y_{N}_{var_key}.npy"), y)
    onp.save(os.path.join(output_dir, f"data_noise_{N}_{var_key}.npy"), noise)

    y = _handle_non_valid_targets(y, bo_method)
    logger.info(f"Generated: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def var_parameters_dict(var_parameters_names, dim=2):
    # Parameters to find
    var_parameters = {}

    # init_length
    if "init_length" in var_parameters_names:
        lb_init_length = 30.0 * onp.ones(dim)
        ub_init_length = 100.0 * onp.ones(dim)
        var_parameters["init_length"] = [lb_init_length, ub_init_length, "uniform"]

    # length
    if "length" in var_parameters_names:
        lb_length = 4.0 * onp.ones(1)
        ub_length = 12.0 * onp.ones(1)
        var_parameters["length"] = [lb_length, ub_length, "uniform"]

    # w
    if "w" in var_parameters_names:
        lb_w = 0.05 * onp.ones(1)  # 0.05
        ub_w = 0.25 * onp.ones(1)  # 0.8
        var_parameters["w"] = [lb_w, ub_w, "uniform"]

    # l_segment
    if "l_segment" in var_parameters_names:
        lb_l_segment = 1.0 * onp.ones(dim)
        ub_l_segment = 15.0 * onp.ones(dim)
        var_parameters["l_segment"] = [lb_l_segment, ub_l_segment, "uniform"]

    # fascicles_length
    if "fascicles_length" in var_parameters_names:
        lb_fascicles_length = 2.0 * onp.ones(
            2 * dim
        )  # 10 # 2.*dim because there are 2 params per ventricle
        ub_fascicles_length = 50.0 * onp.ones(2 * dim)  # 30
        var_parameters["fascicles_length"] = [
            lb_fascicles_length,
            ub_fascicles_length,
            "uniform",
        ]

    # f_angles
    if "fascicles_angles" in var_parameters_names:
        lb_fascicles_angles = (
            -1.0 / 4.0 * onp.pi * np.ones(2 * dim)
        )  # 0.1 # 2.*dim because there are 2 params per ventricle
        ub_fascicles_angles = 3.0 / 4.0 * onp.pi * np.ones(2 * dim)  # 1.57
        var_parameters["fascicles_angles"] = [
            lb_fascicles_angles,
            ub_fascicles_angles,
            "uniform",
        ]

    # branch_angle
    if "branch_angle" in var_parameters_names:
        lb_branch_angle = 5.0 * onp.pi / 180.0 * np.ones(1)
        ub_branch_angle = 45.0 * onp.pi / 180.0 * np.ones(1)
        var_parameters["branch_angle"] = [lb_branch_angle, ub_branch_angle, "uniform"]

    # root_time
    if "root_time" in var_parameters_names:
        lb_root_time = -75.0 * np.ones(1)
        ub_root_time = 50.0 * np.ones(1)
        var_parameters["root_time"] = [lb_root_time, ub_root_time, "uniform"]

    # cv
    if "cv" in var_parameters_names:
        lb_cv = 2.0 * np.ones(1)
        ub_cv = 4.0 * np.ones(1)
        var_parameters["cv"] = [lb_cv, ub_cv, "uniform"]

    return var_parameters


def initial_values(var_parameters_names, patient, meshes_list_pat, myocardium):
    # Initial values for known parameters
    meshes_list = meshes_list_pat
    init_length = 30
    length = 8.0  # [mm]
    w = 0.1
    l_segment = 1.0

    f_len = [20.0, 20.0]
    f_angles = [1.0, 1.0]

    branch_angle = 0.15  # 20. * onp.pi/180. #0.15
    N_it = 20

    # Assign 1. to the parameters to find
    # init_length
    if "init_length" in var_parameters_names:
        init_length_bo = 1.0
    else:
        init_length_bo = init_length

    # length
    if "length" in var_parameters_names:
        length_bo = 1.0
    else:
        length_bo = length  # [mm]

    # w
    if "w" in var_parameters_names:
        w_bo = 1.0
    else:
        w_bo = w

    # l_segment
    if "l_segment" in var_parameters_names:
        l_segment_bo = 1.0
    else:
        l_segment_bo = l_segment  # [mm]

    # fascicles_length
    if "fascicles_length" in var_parameters_names:
        f_len_bo = [1.0, 1.0]
    else:
        f_len_bo = f_len

    # f_angles
    if "fascicles_angles" in var_parameters_names:
        f_angles_bo = [1.0, 1.0]
    else:
        f_angles_bo = f_angles

    # branch_angle
    if "branch_angle" in var_parameters_names:
        branch_angle_bo = 1.0
    else:
        branch_angle_bo = branch_angle

    parameters_values = {
        "patient": patient,
        "meshes_list": meshes_list,
        "init_length": init_length_bo,
        "length": length_bo,
        "w": w_bo,
        "l_segment": l_segment_bo,
        "fascicles_length": f_len_bo,
        "fascicles_angles": f_angles_bo,
        "branch_angle": branch_angle_bo,
        "N_it": N_it,
        "myocardium": myocardium,
    }

    return parameters_values


def optimize_with_BO(
    X,
    y,
    X_star,
    true_x,
    bo_method,
    var_parameters,
    p_x,
    N,
    nIter,
    patient_number,
    output_dir,
    criterion_bo,
    list_variable_params,
):
    logger.info("Starting Bayesian Optimization")

    suffix = f"{N}_{list_variable_params}"

    X_opt_path = os.path.join(output_dir, f"data_X_{suffix}.npy")
    y_opt_path = os.path.join(output_dir, f"data_y_{suffix}.npy")
    info_path = os.path.join(output_dir, f"data_info_{suffix}.pkl")

    if os.path.exists(X_opt_path) and os.path.exists(y_opt_path):
        logger.info("Optimization data already exists. Skipping BO.")
        X = onp.load(X_opt_path)
        y = onp.load(y_opt_path)
        return X, y

    # Configure BO options
    options = {
        "kernel": "Matern12",
        "criterion": criterion_bo,
        "input_prior": p_x,
        "kappa": 2.0,
        "nIter": nIter,
    }

    import time

    t_ini_opt = time.time()
    X, y, info_iterations = bo_method.bo_loop(X, y, X_star, true_x, options)
    t_fin_opt = time.time()

    logger.info(f"Optimization complete in {t_fin_opt - t_ini_opt:.2f} seconds")

    onp.save(X_opt_path, X)
    onp.save(y_opt_path, y)
    with open(info_path, "wb") as f:
        pickle.dump(info_iterations, f)

    logger.info(f"Saved optimized X, y, info to {output_dir}")
    return X, y


def train_gp_model(X, y, options, bo_class, X_star_uniform, gp_state=None):
    if gp_state is None:
        print("Train GP model with valid points...")
        valid = y != bo_class.y_trees_non_valid

        X_valid = X[valid]
        y_valid = y[valid]

        print(f"{len(X_valid)} valid points")

        rng_key = random.PRNGKey(0)
        gp_model = GP(options)

        # Fetch normalized training data
        norm_batch, norm_const = normalize(X_valid, y_valid, bo_class.bounds)

        # Train GP model
        t_ini_train = time.time()
        rng_key = random.split(rng_key)[0]
        opt_params = gp_model.train(norm_batch, rng_key, num_restarts=5)
        t_fin_train = time.time()
        print(f"Training time: {t_fin_train - t_ini_train} s")

        kwargs = {
            "params": opt_params,
            "batch": norm_batch,
            "norm_const": norm_const,
            "bounds": bo_class.bounds,
        }
        #               'kappa': gp_model.options['kappa'],
        #               'gmm_vars': gmm_vars,
        #               'rng_key': rng_key}

        gp_state = [gp_model, [norm_batch, norm_const], kwargs]

    else:
        print("Re-using gp_model (it is not trained again with the new points X, y)...")
        gp_model = gp_state[0]
        norm_batch, norm_const = gp_state[1]
        kwargs = gp_state[2]

    # Compute predicted mean and std
    t_ini_pred = time.time()

    # batches
    n_col = 100
    assert (len(X_star_uniform) / n_col).is_integer(), "Modify n_col"

    reshaped_X = [
        X_star_uniform[i : i + n_col] for i in range(0, len(X_star_uniform), n_col)
    ]

    mean_list, std_list = [], []
    for batch in reshaped_X:
        mean_b, std_b = gp_model.predict(batch, **kwargs)
        mean_list.append(mean_b)
        std_list.append(std_b)
    mean_it = np.concatenate(mean_list)
    std_it = np.concatenate(std_list)

    # # full
    # mean_it, std_it = gp_model.predict(X_star_uniform, **kwargs)

    t_fin_pred = time.time()
    print(f"Predicting time: {t_fin_pred - t_ini_pred} s")

    # Obtain ys and sigmas of X_star_uniform (test points with uniform sampling)
    ys = mean_it * norm_const["sigma_y"] + norm_const["mu_y"]
    sigmas = std_it * norm_const["sigma_y"]

    # Obtain min values predicted by gp model
    X_min = X[np.argmin(y)]
    mean_min, std_min = gp_model.predict(X_min[None, :], **kwargs)

    y_gp_best = mean_min * norm_const["sigma_y"] + norm_const["mu_y"]
    sigma_gp_best = std_min * norm_const["sigma_y"]  # should be low

    return ys, sigmas, y_gp_best, sigma_gp_best, gp_state


def rejection_sampling(ys, sigmas, y_gp_best, sigma_gp_best):
    # Rejection sampling, the likelihood is obtained comparing with best point

    max_likelihood = norm.pdf(
        x=0.0, loc=0.0, scale=np.sqrt(sigma_gp_best**2 + np.min(sigmas) ** 2)
    )

    key = random.PRNGKey(0)  # onp.random.randint(50)
    comparison = random.uniform(key, shape=(ys.shape[0],)) * max_likelihood

    likelihoods = norm.pdf(
        x=0.0, loc=ys - y_gp_best, scale=np.sqrt(sigmas**2 + sigma_gp_best**2)
    )

    accepted_samples = likelihoods > comparison

    print(f"{accepted_samples.sum()} accepted samples")
    return accepted_samples, comparison, likelihoods


def check_accepted_samples(
    N_samples,
    X_star_uniform,
    y_gp_best,
    ys,
    accepted_samples,
    comparison,
    likelihoods,
    var_parameters,
    bo_class,
    tol,
    observed_samples,
    confirmed_samples,
    tree_output_dir,
    patient_number,
    N,
    nIter,
    criterion,
):

    X_accepted = X_star_uniform[accepted_samples]
    ys_accepted = ys[accepted_samples]
    comparison_accepted = comparison[accepted_samples]
    likelihoods_accepted = likelihoods[accepted_samples]

    # Pre-filter samples too far from the best loss
    mask = (ys_accepted - y_gp_best) < 200.0
    X_accepted = X_accepted[mask]
    ys_accepted = ys_accepted[mask]
    comparison_accepted = comparison_accepted[mask]
    likelihoods_accepted = likelihoods_accepted[mask]

    # Sort by likelihood (descending)
    sorted_indices = np.argsort(-likelihoods_accepted)
    X_accepted = X_accepted[sorted_indices]
    ys_accepted = ys_accepted[sorted_indices]
    comparison_accepted = comparison_accepted[sorted_indices]

    # Initialize if first iteration
    if len(confirmed_samples) == 0:
        confirmed_samples["samples_final"] = []
        confirmed_samples["ecg_final"] = []
        confirmed_samples["Tree_final"] = []
        confirmed_samples["loss_final"] = []

    state = []
    observed_samples_new = []
    X_true_new = []
    Y_true_new = []

    def process_candidate(idx, x_accepted, comp):
        result = {
            "idx": idx,
            "x": x_accepted,
            "observed": False,
            "accepted": False,
            "state": "rejected",
        }

        if any(np.array_equal(obs[0], x_accepted) for obs in observed_samples):
            result["observed"] = True
            return result

        try:
            ecg_i, endo_i, LVtree_i, RVtree_i = bo_class.update_purkinje_tree(
                np.array([x_accepted]), 1.0, var_parameters
            )
        except Exception:
            return result

        loss_i, ind_loss_i = bo_class.calculate_loss(
            ecg_i, cross_correlation=True, return_ind=True
        )
        y_true = loss_i
        delta = y_true - y_gp_best

        result.update(
            {
                "y": y_true,
                "ecg": ecg_i,
                "endo": endo_i,
                "trees": [LVtree_i, RVtree_i],
                "ind_loss": ind_loss_i,
                "accepted": delta < tol,
                "state": "accepted" if delta < tol else "rejected",
            }
        )

        return result

    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for idx, (x_acc, comp) in enumerate(zip(X_accepted, comparison_accepted)):
            futures.append(executor.submit(process_candidate, idx, x_acc, comp))

        results = [f.result() for f in as_completed(futures)]

    # Sort back by original index
    results.sort(key=lambda r: r["idx"])

    for result in results:
        x = result["x"]
        if result["observed"]:
            continue

        y = result.get("y", bo_class.y_trees_non_valid)
        X_true_new.append(x)
        Y_true_new.append(y)

        obs_value = "non_valid" if not result["accepted"] else result["ecg"]
        observed_samples_new.append([x, obs_value])

        state.append(result["state"])

        if result["accepted"]:
            idx_final = len(confirmed_samples["samples_final"])
            confirmed_samples["samples_final"].append(x)
            confirmed_samples["ecg_final"].append([result["ecg"], result["ind_loss"]])
            confirmed_samples["Tree_final"].append(result["trees"])
            confirmed_samples["loss_final"].append(y)

            LV_path = os.path.join(
                tree_output_dir,
                f"LVtree_N{N}_nIter{nIter}_criterion{criterion}_{idx_final}.vtu",
            )
            RV_path = os.path.join(
                tree_output_dir,
                f"RVtree_N{N}_nIter{nIter}_criterion{criterion}_{idx_final}.vtu",
            )
            EN_path = os.path.join(
                tree_output_dir,
                f"propeiko_N{N}_nIter{nIter}_criterion{criterion}_{idx_final}.vtu",
            )

            result["trees"][0].save(LV_path)
            result["trees"][1].save(RV_path)
            result["endo"].save_pv(EN_path)

            if len(confirmed_samples["samples_final"]) >= N_samples:
                logger.info("Accepted samples goal reached.")
                return "ok", confirmed_samples

        if len(state) >= 100 and state[-100:] == ["rejected"] * 100:
            logger.warning("100 consecutive rejections — triggering GP retrain.")
            return "retrain gp", {
                "X_true_new": X_true_new,
                "Y_true_new": Y_true_new,
                "Observed_samples": observed_samples + observed_samples_new,
                "Confirmed_samples": confirmed_samples,
            }

    logger.info("All candidates processed.")
    return "retrain gp", {
        "X_true_new": X_true_new,
        "Y_true_new": Y_true_new,
        "Observed_samples": observed_samples + observed_samples_new,
        "Confirmed_samples": confirmed_samples,
    }


def run_rejection_sampling_loop(
    X,
    y,
    bo_method,
    var_parameters,
    N_samples,
    patient_number,
    output_dir,
    list_variable_params,
    criterion_bo,
    tol=100.0,
    nn=500_000,
    max_rejections=50,
    resume=False,
):
    logger.info("Starting rejection sampling loop")

    suffix = f"{N_samples}_{list_variable_params}"
    X_save_path = os.path.join(output_dir, f"rejection_X_{suffix}.npy")
    y_save_path = os.path.join(output_dir, f"rejection_y_{suffix}.npy")
    samples_save_path = os.path.join(output_dir, f"rejection_samples_{suffix}.pkl")

    # Carga si se quiere reanudar
    if resume and os.path.exists(samples_save_path):
        logger.info("Resuming from previous confirmed samples...")
        with open(samples_save_path, "rb") as f:
            confirmed_samples = pickle.load(f)
        obs_samples = confirmed_samples.get("observed", [])
        conf_samples = confirmed_samples.get("confirmed", {})
    else:
        obs_samples = []
        conf_samples = {}

    folder_trees = f"/Trees_N{N_samples}_criterion{criterion_bo}_{list_variable_params}"
    tree_output_dir = os.path.join(output_dir, folder_trees)
    if os.path.exists(tree_output_dir):
        shutil.rmtree(tree_output_dir)
    os.makedirs(tree_output_dir)

    gp_rejection = None
    rejection_n = 1

    while True:
        logger.info(f"Rejection iteration {rejection_n}")
        key = random.PRNGKey(rejection_n)

        # y_best = np.min(y)
        X_star_uni = bo_method.lb_params + (
            bo_method.ub_params - bo_method.lb_params
        ) * random.uniform(key, shape=(nn, bo_method.dim))

        ys, sigmas, y_gp_best, sigma_gp_best, gp_rejection = train_gp_model(
            X,
            y,
            {
                "kernel": "Matern12",
                "criterion": criterion_bo,
                "input_prior": None,
                "kappa": 2.0,
                "nIter": 0,  # no importa aquí
            },
            bo_class=bo_method,
            X_star_uniform=X_star_uni,
            gp_state=gp_rejection,
        )

        accepted_samples, comparison, likelihoods = rejection_sampling(
            ys, sigmas, y_gp_best, sigma_gp_best
        )

        state_final, info_final = check_accepted_samples(
            N_samples=N_samples,
            X_star_uniform=X_star_uni,
            y_gp_best=y_gp_best,
            ys=ys,
            accepted_samples=accepted_samples,
            comparison=comparison,
            likelihoods=likelihoods,
            var_parameters=var_parameters,
            bo_class=bo_method,
            tol=tol,
            observed_samples=obs_samples,
            confirmed_samples=conf_samples,
            tree_output_dir=tree_output_dir,
            patient_number=patient_number,
            N=N_samples,
            nIter=100,
            criterion=criterion_bo,
        )

        if state_final == "ok":
            logger.info("Sampling loop completed successfully")
            onp.save(X_save_path, X)
            onp.save(y_save_path, y)
            with open(samples_save_path, "wb") as f:
                pickle.dump(conf_samples, f)
            return conf_samples  # Puede incluir trees, ECGs, losses, etc.

        elif state_final == "retrain gp":
            obs_samples = info_final["Observed_samples"]
            conf_samples = info_final["Confirmed_samples"]
            X_new = info_final["X_true_new"]
            y_new = onp.asarray(info_final["Y_true_new"])
            X = onp.concatenate([X, X_new], axis=0)
            y = onp.concatenate([y, y_new], axis=0)

            logger.info(f"Retraining GP with {len(X_new)} new samples")
            onp.save(X_save_path, X)
            onp.save(y_save_path, y)
            with open(samples_save_path, "wb") as f:
                pickle.dump({"observed": obs_samples, "confirmed": conf_samples}, f)

        if rejection_n >= max_rejections:
            logger.error(
                f"Reached max rejection iterations ({max_rejections}) without success"
            )
            raise RuntimeError(
                f"Could not accept {N_samples} samples after {max_rejections} attempts"
            )

        rejection_n += 1


def main():
    patient_number = "demo"
    N = 250
    nIter = 100
    mode = "load"  # or "generate"
    criterion_bo = "EI"
    var_parameters_list = [
        "init_length",
        "fascicles_length",
        "fascicles_angles",
        "root_time",
        "cv",
    ]

    # Step 1: Load patient mesh & data
    patient, meshes_list_pat, myo, output_dir = prepare_patient(patient_number)

    # Step 2: Set up BO method and parameter space
    dim = 2
    var_parameters = var_parameters_dict(var_parameters_list, dim)
    initial_params = initial_values(var_parameters_list, patient, meshes_list_pat, myo)

    Tree_bo = BO_PurkinjeTree(**initial_params)
    bo_method = BO_ecg(Tree_bo)

    # Step 3: Define MSE function and input prior
    f, p_x = bo_method.mse_jaxbo(ground_truth=None, variable_parameters=var_parameters)

    # Step 4: Load or generate initial training data
    try:
        if mode == "load":
            X, y = load_initial_data(N, output_dir, var_parameters, bo_method)
        else:
            X, y = generate_initial_data(N, output_dir, var_parameters, bo_method)
    except FileNotFoundError as e:
        logger.warning(f"{e} — falling back to generation.")
        X, y = generate_initial_data(N, output_dir, var_parameters, bo_method)

    # Step 5: Set test data (X_star)
    if bo_method.dim == 2:
        X_star, _, _ = bo_method.set_test_data()
    else:
        X_star = bo_method.set_test_data()

    # Step 6: Load ground truth parameters
    expected_dim = X_star.shape[1] if len(X_star.shape) == 2 else bo_method.dim
    true_x = load_ground_truth_parameters(patient_number, expected_dim)

    # Step 7: Run Bayesian Optimization (with checkpointing)
    list_variable_params = "_".join(list(var_parameters.keys()))

    X, y = optimize_with_BO(
        X,
        y,
        X_star,
        true_x,
        bo_method,
        var_parameters,
        p_x,
        N,
        nIter,
        patient_number,
        output_dir,
        criterion_bo,
        list_variable_params,
    )

    logger.info("BO phase complete. Ready to start sampling/rejection loop.")

    # Step 8: Rejection Sampling to find N_samples good candidates
    N_samples = 30
    confirmed_samples = run_rejection_sampling_loop(
        X=X,
        y=y,
        bo_method=bo_method,
        var_parameters=var_parameters,
        N_samples=N_samples,
        patient_number=patient_number,
        output_dir=output_dir,
        list_variable_params=list_variable_params,
        criterion_bo=criterion_bo,
        tol=100.0,
        nn=500000,
        max_rejections=50,
        resume=False,
    )
    logger.info(
        f"Rejection loop completed with {len(confirmed_samples['samples_final'])} accepted samples."
    )


if __name__ == "__main__":
    main()
