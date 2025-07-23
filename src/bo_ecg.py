import logging
import traceback

import numpy as onp
import jax.numpy as np
from jax import random, vmap

import matplotlib.pyplot as plt

from pyDOE import lhs

from dataclasses import dataclass
from enum import Enum
from typing import List

from jaxbo.input_priors import uniform_prior, gaussian_prior
from jaxbo.models import GP
from jaxbo.utils import normalize, compute_w_gmm

from bo_purkinje_tree import BO_PurkinjeTree

onp.random.seed(1234)
logger = logging.getLogger(__name__)

class PriorType(str, Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"

@dataclass
class OptimParam:
    parameter: str
    lower: np.ndarray
    upper: np.ndarray
    prior: PriorType

class BO_ecg:

    def __init__(self, bo_purkinje_tree: BO_PurkinjeTree):
        self.bo_purkinje_tree = bo_purkinje_tree


    def plot_ecg_match(
        self,
        predicted: np.ndarray,
        filename_match: str | None = None
    ) -> None:
        """
        Plot the best alignment between predicted and ground-truth ECG leads,
        based on delay calculated by cross-correlation loss.

        Parameters
        ----------
        predicted : np.ndarray
            Structured array with predicted ECG for each lead (same format as ground_truth).
        filename_match : str or None
            If provided, saves the plot to this filename + '_ecg_match.pdf'.
        """
        # Find shift and plot best match
        try:
            _, ind_min_loss = self.calculate_loss(
                predicted, cross_correlation=True, return_ind=True
            )
        except Exception as e:
            logger.error(f"Failed to compute cross-correlation loss: {e}")
            return

        len_gt = self.ground_truth.shape[0]
        delay = len_gt - 1 + ind_min_loss

        logger.info(f"Best alignment found with delay index: {ind_min_loss} (total shift: {delay})")
        
        fig, axs = plt.subplots(
            3, 4, figsize=(10, 13), dpi=120, sharex=True, sharey=True
        )

        axs[0, 0].legend(["Ground truth", "Predicted"], fontsize=8)

        for ax, lead_name in zip(axs.ravel(), self.ground_truth.dtype.names):
            try:
                t_gt = delay - len_gt + 1 + np.arange(len_gt)

                ax.plot(
                    t_gt,
                    self.ground_truth[lead_name],
                    color="tab:blue",
                    alpha=0.6,
                    label="Ground truth",
                )
                ax.plot(
                    predicted[lead_name],
                    color="tab:red",
                    alpha=0.6,
                    label="Predicted",
                )
                ax.axvspan(t_gt[0], t_gt[-1], alpha=0.2, color="wheat")

                ax.grid(linestyle="--", alpha=0.4)
                ax.set_title(lead_name)

            except Exception as e:
                logger.warning(f"Failed to plot lead {lead_name}: {e}")
                continue

        fig.tight_layout()

        if filename_match:
            output_file = f"{filename_match}_ecg_match.pdf"
            fig.savefig(output_file)
            logger.info(f"ECG match plot saved to: {output_file}")


    def _extract_overlapping_section(self, ground_truth, predicted, delay):
        """
        Extract overlapping sections of ground_truth and predicted ECG signals based on a delay.

        Parameters
        ----------
        ground_truth : np.ndarray
            ECG signal (1D array), fixed in time.
        predicted : np.ndarray
            ECG signal to be shifted by `delay` relative to ground_truth.
        delay : int
            Shift value, consistent with np.correlate(..., mode="full").
            Positive delay → predicted is delayed (shifted right).
            Negative delay → predicted is advanced (shifted left).

        Returns
        -------
        tuple of np.ndarray
            (gt_section, pred_section): aligned overlapping segments.

        Raises
        ------
        ValueError:
            If overlapping segment length is zero or mismatch occurs.
        """

        len_gt = ground_truth.shape[0]
        len_pred = predicted.shape[0]

        shift = delay
        start_gt = max(0, shift)
        end_gt = min(len_gt, len_pred + shift)

        start_pred = max(0, -shift)
        end_pred = start_pred + (end_gt - start_gt)

        overlap_len = end_gt - start_gt

        if overlap_len <= 0:
            logger.warning(
                f"Invalid overlap length: ground_truth[{start_gt}:{end_gt}], "
                f"predicted[{start_pred}:{end_pred}] (delay={delay})"
            )
            raise ValueError("No overlapping region found with given delay.")

        gt_section = ground_truth[start_gt:end_gt]
        pred_section = predicted[start_pred:end_pred]

        if gt_section.shape[0] != pred_section.shape[0]:
            logger.error(
                f"Mismatch in overlapping lengths: ground_truth={gt_section.shape[0]}, "
                f"predicted={pred_section.shape[0]} (delay={delay})"
            )
            raise ValueError("Mismatch in length of overlapping ECG sections.")

        return gt_section, pred_section


    def _loss_cross_correlation(self, predicted, ground_truth, return_ind):
        """
        Computes the cross-correlation loss between predicted and ground truth ECG signals by shifting the predicted signal
        and calculating the mean squared error (MSE) for each lead at each shift. The function finds the shift that yields
        the minimum total loss across all leads.

        Parameters
        ----------
        predicted : np.ndarray
            The predicted ECG signal, structured as a numpy array with named leads.
        ground_truth : np.ndarray
            The ground truth ECG signal, structured as a numpy array with named leads.
        return_ind : bool
            If True, returns both the minimum loss and the corresponding shift index. If False, returns only the minimum loss.
        Returns
        -------
        min_loss : float
            The minimum cross-correlation loss found across all shifts.
        best_shift : int, optional
            The shift index corresponding to the minimum loss (returned only if `return_ind` is True).
        Notes
        -----
        - For each possible shift, the function extracts overlapping sections of the predicted and ground truth signals,
          computes the MSE for each lead, and sums the losses across all leads.
        """
        len_gt = ground_truth.shape[0]
        len_pred = predicted.shape[0]

        min_shift = len_gt - 1
        max_shift = len_pred - 1
        shift_range = np.arange(min_shift, max_shift + 1)

        loss_by_shift = []

        for shift in shift_range:
            lead_losses = []
            for lead in ground_truth.dtype.names:
                try:
                    gt_seg, pred_seg = self._extract_overlapping_section(
                        ground_truth[lead], predicted[lead], int(shift)
                    )
                    mse = np.mean((gt_seg - pred_seg) ** 2)
                    lead_losses.append(mse)
                except Exception as e:
                    logger.warning(f"Failed to compute loss for lead {lead} at shift {shift}: {e}")
                    lead_losses.append(np.inf)

            total_loss = np.sum(lead_losses)
            loss_by_shift.append(total_loss)

        loss_array = np.array(loss_by_shift)
        idx_min = int(np.argmin(loss_array))
        min_loss = float(loss_array[idx_min])
        best_shift = int(shift_range[idx_min])

        logger.info(f"Best cross-correlated loss: {min_loss:.4e} at shift index {best_shift}")

        if return_ind:
            return min_loss, best_shift
        return min_loss


    def _loss_direct(self, predicted, ground_truth):
        """
        Computes the total mean squared error (MSE) loss between predicted and ground truth ECG signals for each lead.

        Parameters
        ----------
        predicted : np.ndarray or structured array
            Predicted ECG signals, structured with named leads.
        ground_truth : np.ndarray or structured array
            Ground truth ECG signals, structured with named leads.

        Returns
        ----------
            The total MSE loss summed across all leads. If a lead fails to compute, adds infinity to the loss.

        """
        total_loss = 0.0
        for lead in ground_truth.dtype.names:
            try:
                gt = ground_truth[lead]
                pred = predicted[lead]
                trim_len = min(gt.shape[0], pred.shape[0])
                mse = np.mean((gt[:trim_len] - pred[:trim_len]) ** 2)
                total_loss += mse
            except Exception as e:
                logger.warning(f"Failed to compute direct MSE for lead {lead}: {e}")
                total_loss += np.inf

        logger.info(f"Direct MSE loss: {total_loss:.4e}")
        return float(total_loss)


    def calculate_loss(self, predicted, cross_correlation=True, return_ind=False, ecg_pat=None):
        """
        Compute the ECG loss between predicted and ground truth, with or without temporal alignment.

        Parameters
        ----------
        predicted : np.ndarray
            Structured array of predicted ECG leads.
        cross_correlation : bool
            If True, aligns signals before computing MSE using delay shift.
        return_ind : bool
            If True, returns the delay index with minimum loss.
        ecg_pat : np.ndarray or None
            Optional ECG ground truth to override self.ground_truth.

        Returns
        -------
        float or (float, int)
            If return_ind=False: minimum loss.
            If return_ind=True: (min_loss, delay_index).
        """
        ground_truth = ecg_pat if ecg_pat is not None else self.ground_truth

        if cross_correlation:
            return self._loss_cross_correlation(predicted, ground_truth, return_ind)
        else:
            return self._loss_direct(predicted, ground_truth)


    def _build_bounds_and_prior(self, variable_parameters: List[OptimParam]):
        """
        Extract bounds and prior distribution from a dictionary of OptimParam.

        Parameters
        ----------
        variable_parameters : List[OptimParam]
            List of OptimParam objects.

        Returns
        -------
        lb_params : np.ndarray
        ub_params : np.ndarray
        p_x_params : callable
        """
        lb_list = []
        ub_list = []
        prior_types = []

        for param in variable_parameters:
            lb_list.extend(param.lower.tolist())
            ub_list.extend(param.upper.tolist())
            prior_types.append(param.prior)

        lb_params = np.array(lb_list)
        ub_params = np.array(ub_list)

        if len(set(prior_types)) != 1:
            raise ValueError("All parameters must use the same prior distribution.")

        dist_type = prior_types[0]
        if dist_type == PriorType.UNIFORM:
            p_x_params = uniform_prior(lb_params, ub_params)
        elif dist_type == PriorType.GAUSSIAN:
            p_x_params = gaussian_prior(lb_params, ub_params)
        else:
            raise NotImplementedError(f"Unsupported prior type: {dist_type}")

        logger.info(f"Bounds set for {len(lb_params)} parameters using {dist_type.value} prior.")
        return lb_params, ub_params, p_x_params


    def mse_jaxbo(
        self,
        ground_truth: np.ndarray,
        variable_parameters: list[OptimParam],
    ) -> tuple[callable, callable, dict]:
        """
        Configure the optimization objective and priors from ECG and variable definitions.

        Parameters
        ----------
        ground_truth : np.ndarray
            Structured ECG array used as reference.
        variable_parameters : list[OptimParam]
            List of OptimParam objects defining variable names and their optimization settings.

        Returns
        -------
        f : callable
            Objective function f(x) → MSE or penalty.
        p_x_params : callable
            Prior distribution over input space.
        bounds : dict
            {"lb": array, "ub": array}
        """
        self.ground_truth = ground_truth
        self.variable_parameters = variable_parameters

        lb_params, ub_params, p_x_params = self._build_bounds_and_prior(variable_parameters)
        bounds = {"lb": lb_params, "ub": ub_params}
        dim = len(lb_params)

        def f(x):
            try:
                x = x.astype(float)
                var_dict = self.set_dictionary_variables(variable_parameters, x)
                predicted, *_ = self.bo_purkinje_tree.run_ECG(
                    n_sim=0, modify=True, side="both", **var_dict
                )
                loss = self.calculate_loss(predicted)
                logger.info(f"Evaluated f(x): loss={loss:.4e}, x={x}")
            except Exception as e:
                logger.error(f"Simulation failed in f(x): {e}")
                traceback.print_exc()
                loss = getattr(self, "y_trees_non_valid", 1e6)  # Fallback penalty
            return loss

        self.f = f
        self.p_x_params = p_x_params
        self.bounds = bounds
        self.lb_params = lb_params
        self.ub_params = ub_params
        self.dim = dim
        self.bounds = {"lb": lb_params, "ub": ub_params}
        return f, p_x_params, bounds


    def set_initial_training_data(
        self,
        N: int,
        noise: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate initial training data using Latin Hypercube Sampling.

        Parameters
        ----------
        N : int
            Number of initial training samples.
        noise : float
            Proportional noise level to add to output values (default: 0.0).

        Returns
        -------
        X : np.ndarray
            Sampled input points (N × dim).
        y : np.ndarray
            Corresponding noisy output values (N × 1).
        """
        
        if not hasattr(self, "f") or not callable(self.f):
            raise RuntimeError("Objective function f(x) is not defined. Call mse_jaxbo() first.")
        if not hasattr(self, "lb_params") or not hasattr(self, "ub_params"):
            raise RuntimeError("Parameter bounds not defined. Call mse_jaxbo() first.")
        
        self.noise = noise

        logger.info(f"Generating {N} initial samples with noise level {self.noise}")

        span = self.ub_params - self.lb_params
        X = self.lb_params + span * lhs(self.dim, N)

        logger.info(f"Evaluating objective function on {N} samples...")
        y = np.array([self.f(x) for x in X])

        if self.noise > 0.0:
            std_y = y.std()
            noise_term = self.noise * std_y * onp.random.normal(size=y.shape)
            y += noise_term
            logger.info(f"Added Gaussian noise to outputs (std: {std_y:.3e})")

        return X, y


    def set_test_data(self) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate test input data (X_star) depending on dimensionality for BO visualization or acquisition.

        Returns
        -------
        X_star : np.ndarray
            Test input points for acquisition.
        If dim == 2:
            Also returns XX, YY meshgrid for surface visualization.
        """

        if not hasattr(self, "dim") or not hasattr(self, "lb_params") or not hasattr(self, "ub_params"):
            raise RuntimeError("Bounds and dimensionality must be set by mse_jaxbo() before calling set_test_data().")

        dim = self.dim
        
        if dim == 1:
            nn = 1000
            logger.info(f"Generating 1D test data with {nn} points")
            x = np.linspace(self.lb_params[0], self.ub_params[0], nn)
            X_star = x[:, None]
            return X_star

        elif dim == 2:
            nn = 10
            logger.info(f"Generating 2D test grid with {nn}×{nn} = {nn*nn} points")
            xx = np.linspace(self.lb_params[0], self.ub_params[0], nn)
            yy = np.linspace(self.lb_params[1], self.ub_params[1], nn)
            XX, YY = np.meshgrid(xx, yy)
            X_star = np.column_stack([XX.ravel(), YY.ravel()])
            return X_star, XX, YY

        else:
            nn = 25000
            logger.info(f"Generating {nn} random test points in {dim}D using LHS")
            span = self.ub_params - self.lb_params
            X_star = self.lb_params + span * lhs(dim, nn)
            return X_star


    def plot_mse(
        self,
        X: np.ndarray,
        y: np.ndarray,
        N: int,
        file_name: str,
    ) -> None:
        """
        Plot MSE values over Bayesian Optimization iterations.

        Parameters
        ----------
        X : np.ndarray
            Input samples.
        y : np.ndarray
            Corresponding MSE values.
        N : int
            Number of initial training samples (used to offset BO iterations).
        file_name : str
            Path prefix where the plot will be saved (PDF).
        """

        if not hasattr(self, "nIter"):
            raise RuntimeError("Missing self.nIter — ensure bo_loop() has run first.")
        if not hasattr(self, "variable_parameters"):
            raise RuntimeError("Missing self.variable_parameters — ensure mse_jaxbo() was called.")
        
        idx_best = np.argmin(y)
        best_x = onp.array(X[idx_best, :])

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.nIter) + 1, y[N:])
        ax.axhline(y=np.min(y), color="r", linestyle="--", alpha=0.6)

        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("mse")
        ax.set_title("MSE over BO Iterations")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # Text box with parameter summary
        param_names = [p.param for p in self.variable_parameters]
        best_x_fmt = [f"{v:.2f}" for v in best_x]
        y_min_val = float(np.min(y))

        if idx_best >= N:
            best_y_it = f"(in iteration {idx_best - N + 1})"
        else:
            best_y_it = "(in training points)"

        summary_text = (
            f"params: {param_names}\n"
            f"best_x: {best_x_fmt}\n"
            f"y_min: {y_min_val:.2e} {best_y_it}"
        )

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox=props,
        )

        ax.legend(fontsize=8)
        fig.tight_layout()

        output_path = f"{file_name}_MSE.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"MSE plot saved to: {output_path}")


    def update_purkinje_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_parameters: dict[str, OptimParam],
    ) -> tuple[np.ndarray, object, object]:
        """
        Update the Purkinje tree using the best parameters from BO and return the resulting ECG and trees.

        Parameters
        ----------
        X : np.ndarray
            Candidate input parameter values (N × dim).
        y : np.ndarray
            Corresponding objective function values (MSE).
        variable_parameters : dict[str, OptimParam]
            Mapping of parameter names to bounds/prior configuration.

        Returns
        -------
        ecg_bo : np.ndarray
            The simulated ECG using the best parameters found.
        LVtree_bo : object
            The updated left ventricular tree.
        RVtree_bo : object
            The updated right ventricular tree.
        """
        # Update tree with optimal parameters
        idx_best = onp.argmin(y)
        best_x = X[idx_best]
        
        logger.info(f"Updating Purkinje tree with best_x at index {idx_best}: {best_x}")

        param_dict = self.set_dictionary_variables(
            var_parameters=variable_parameters,
            x_values=best_x,
        )

        ecg_bo, LVtree_bo, RVtree_bo = self.bo_purkinje_tree.run_ECG(
            modify=True,
            side="both",
            **param_dict,
        )

        return ecg_bo, LVtree_bo, RVtree_bo


    def set_dictionary_variables(
        self,
        var_parameters: list[OptimParam],
        x_values: np.ndarray
    ) -> dict[str, object]:
        """
        Reconstruct the dictionary of parameters expected by run_ECG from a flat array of values.

        Parameters
        ----------
        var_parameters : dict[str, OptimParam]
            Dictionary of parameter names and their optimization settings.
        x_values : np.ndarray
            Flat array of parameter values sampled from the search space.

        Returns
        -------
        dict_parameters : dict
            Structured dictionary to be passed to run_ECG.
        """
        dict_parameters = {}
        ind = 0
        x_values = np.array(x_values, dtype=float)

        for param in var_parameters:
            var_name = param.parameter
            param_dim = param.lower.size

            if var_name in {"fascicles_length", "fascicles_angles"} and param_dim == 4:
                # Interpret as 2x2 nested list
                dict_parameters[var_name] = [
                    [x_values[ind], x_values[ind + 1]],
                    [x_values[ind + 2], x_values[ind + 3]],
                ]
                ind += 4

            elif var_name in {"w", "branch_angle", "length"} and param_dim == 1:
                # Duplicate value in a 2-element list (for symmetry?)
                value = x_values[ind]
                dict_parameters[var_name] = [value, value]
                ind += 1

            elif var_name in {"root_time", "cv"} and param_dim == 1:
                dict_parameters[var_name] = x_values[ind]
                ind += 1

            elif param_dim == 2:
                dict_parameters[var_name] = [
                    x_values[ind],
                    x_values[ind + 1],
                ]
                ind += 2

            else:
                # Default: assign as flat array of expected shape
                dict_parameters[var_name] = x_values[ind : ind + param_dim]
                ind += param_dim

        if ind != len(x_values):
                logger.warning(
                    f"Total consumed parameters ({ind}) does not match x_values length ({len(x_values)}). "
                    f"Check var_parameters definitions."
                )

        return dict_parameters


    def bo_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_star: np.ndarray,
        true_x: np.ndarray | None,
        options: dict,
        save_info: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Run the Bayesian Optimization loop using a GP model and acquisition function.

        Parameters
        ----------
        X : np.ndarray
            Initial training input data (N × dim).
        y : np.ndarray
            Corresponding training output values.
        X_star : np.ndarray
            Test points where acquisition and predictions will be evaluated.
        true_x : np.ndarray or None
            Ground-truth solution (used only for logging).
        options : dict
            BO configuration (must include 'nIter' and 'criterion').
        save_info : bool
            If True, stores predicted mean, std, acquisition values and weights for each iteration.

        Returns
        -------
        X : np.ndarray
            Final input data after BO iterations.
        y : np.ndarray
            Final output values.
        info_iterations : list
            [means, stds, weights, acquisition values] — only if save_info=True.
        """

        gp_model = GP(options)
        rng_key = random.PRNGKey(0)

        mean_iterations = []
        std_iterations = []
        w_pred_iterations = []
        a_pred_iterations = []

        n_iter = options["nIter"]
        self.nIter = n_iter

        for it in range(n_iter):
            logger.info(f"--- BO Iteration {it + 1}/{n_iter} ---")

            # Normalize current batch
            norm_batch, norm_const = normalize(X, y, self.bounds)

            # Train GP
            logger.info("Training GP model...")
            rng_key = random.split(rng_key)[0]
            opt_params = gp_model.train(norm_batch, rng_key, num_restarts=5)

            # Fit GMM if needed
            criterion = options.get("criterion", "")
            if criterion in {"LW-LCB", "LW-US"}:
                logger.info("Fitting GMM on normalized data...")
                rng_key = random.split(rng_key)[0]
                gmm_vars = gp_model.fit_gmm(
                    params=opt_params,
                    batch=norm_batch,
                    norm_const=norm_const,
                    bounds=self.bounds,
                    kappa=gp_model.options["kappa"],
                    rng_key=rng_key,
                    N_samples=10000,
                )
                logger.info("GMM fitted successfully.")
            else:
                gmm_vars = None
                logger.info("No GMM fitted.")

            kwargs = {
                "params": opt_params,
                "batch": norm_batch,
                "norm_const": norm_const,
                "bounds": self.bounds,
                "kappa": gp_model.options["kappa"],
                "gmm_vars": gmm_vars,
                "rng_key": rng_key,
            }

            if save_info:
                logger.info("Computing predictions and acquisition...")
                mean_it, std_it = gp_model.predict(X_star, **kwargs)
                y_it = mean_it * norm_const["sigma_y"] + norm_const["mu_y"]
                sigma_it = std_it * norm_const["sigma_y"]
                mean_iterations.append(y_it)
                std_iterations.append(sigma_it)

                if criterion in {"LW-LCB", "LW-US"}:
                    w_pred_it = compute_w_gmm(X_star, **kwargs)
                else:
                    w_pred_it = np.zeros(X_star.shape[0])
                w_pred_iterations.append(w_pred_it)

                a_pred_it = vmap(lambda x: gp_model.acquisition(x, **kwargs))(X_star)
                a_pred_iterations.append(a_pred_it)

            # Evaluate training error
            train_preds = gp_model.predict(X, **kwargs)[0]
            train_error = np.mean((train_preds - norm_batch["y"]) ** 2)
            logger.info(f"Training error: {train_error:.4e}")

            # Compute next acquisition point
            logger.info("Computing next acquisition point...")
            new_X, _, _ = gp_model.compute_next_point_lbfgs(
                num_restarts=50, **kwargs
            )

            # Evaluate new points
            logger.info("Evaluating new point(s)...")
            new_y = np.array([self.f(x) for x in new_X])
            new_y += self.noise * new_y.std() * onp.random.normal(size=new_y.shape)

            # Update training set
            logger.info("Augmenting dataset...")
            X = np.vstack([X, new_X])
            y = np.concatenate([y, new_y])

            # Report current best
            idx_best = int(np.argmin(y))
            best_x = X[idx_best]
            best_y = y[idx_best]

            logger.info(f"Best x so far: {best_x}")
            logger.info(f"Best y: {best_y:.4e}")
            if true_x is not None:
                logger.info(f"True x: {true_x}")

        info_iterations = [
            mean_iterations,
            std_iterations,
            w_pred_iterations,
            a_pred_iterations,
        ] if save_info else []

        return X, y, info_iterations
