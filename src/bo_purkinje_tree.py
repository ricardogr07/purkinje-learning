import os
import logging
from dataclasses import dataclass, asdict, is_dataclass
from typing import List, Any
from enum import Enum

import numpy as onp

from purkinje_uv import PurkinjeTree, FractalTree, Parameters
from myocardial_mesh import MyocardialMesh

logger = logging.getLogger(__name__)


class MeshSuffix(str, Enum):
    LV = "LVendo"
    RV = "RVendo"


@dataclass
class BO_PurkinjeTreeConfig:
    """
    Configuration parameters for generating a Purkinje tree for ECG simulation.

    Attributes:
        init_length (float): Initial segment length at the root of the tree [mm].
        length (float): Total intended length of Purkinje branches [mm].
        w (float): Controls the angular spread (width) of branches.
        l_segment (float): Length of each branch segment [mm].
        fascicles_length (List[float]): Lengths of predefined fascicles (per bundle).
        fascicles_angles (List[float]): Angles (in radians) for each fascicle direction.
        branch_angle (float): Bifurcation angle at each branching point [radians].
        N_it (int): Number of growth iterations for the tree.
        save_pmjs (bool): Whether to save Purkinje-myocardial junction locations to file.
        kmax (int): Maximum number of iterations for ECG convergence.
    """

    init_length: float
    length: float
    w: float
    l_segment: float
    fascicles_length: List[float]
    fascicles_angles: List[float]
    branch_angle: float
    N_it: int
    save_pmjs: bool = False
    kmax: int = 8


class BO_PurkinjeTree:
    """
    Class to perform Bayesian Optimization on a Purkinje tree model for cardiac simulations.ee structures within a myocardial mesh, using Bayesian Optimization to tune tree parameters. It provides methods for initializing fractal trees for the left and right ventricles, applying parameter modifications, coupling Purkinje and myocardial activation, running ECG simulations, and saving tree structures.

    Attributes:
        meshes_list (List[Any]): List containing LV/RV node and mesh geometry references.
        LVfractaltree (FractalTree): Fractal tree instance for the left ventricle.
        RVfractaltree (FractalTree): Fractal tree instance for the right ventricle.
        Various configuration parameters unpacked from BO_PurkinjeTreeConfig.

    Methods:
        run_ECG(LVfractaltree=None, RVfractaltree=None, n_sim=0, modify=False, side='both', **kwargs):
            Runs the ECG simulation, optionally modifying tree parameters, and returns the simulated ECG and tree objects.
        save_fractaltrees(filename_LVtree, filename_RVtree):
            Saves the current LV and RV fractal tree structures to files.
    """

    def __init__(
        self,
        patient: str,
        meshes_list: List[Any],
        config: BO_PurkinjeTreeConfig,
        myocardium: MyocardialMesh,
    ):
        """
        Initializes the Bayesian Optimization class for the Purkinje tree generation.

        Args:
            patient (str): Patient identifier.
            meshes_list (List): LV/RV node and mesh geometry references.
            config (BO_PurkinjeTreeConfig): Configuration parameters for Purkinje growth.
            myocardium (MyocardialMesh): Myocardial geometry and simulation domain.
        """
        self.patient = patient
        self.meshes_list = meshes_list
        self.myocardium = myocardium

        # Unpack config attributes
        if is_dataclass(config):
            for key, value in asdict(config).items():
                setattr(self, key, value)

        # Log config values
        scalar_config = {
            k: v for k, v in asdict(config).items() if isinstance(v, (float, int, bool))
        }

        logger.debug(f"BO_PurkinjeTree initialized with config: {scalar_config}")

        logger.info(f"Initializing BO_Purkinje for patient {self.patient}")
        self.LVfractaltree, self.RVfractaltree = self._initialize()

    def _build_fractal_tree_parameters(
        self, mesh_suffix: MeshSuffix, node_ids: tuple[int, int]
    ) -> Parameters:
        """
        Creates and configures a Parameters instance for a Fractal Tree.

        Args:
            mesh_suffix (MeshSuffix): Either MeshSuffix.LV or MeshSuffix.RV
            node_ids (tuple): (init_node_id, second_node_id)

        Returns:
            Parameters: Configured Parameters object
        """
        param = Parameters()
        param.meshfile = f"{self.patient}_{mesh_suffix}_heart_cut.obj"
        param.init_node_id, param.second_node_id = node_ids
        param.init_length = self.init_length
        param.length = self.length
        param.w = self.w
        param.l_segment = self.l_segment
        param.fascicles_length = self.fascicles_length
        param.fascicles_angles = self.fascicles_angles
        param.branch_angle = self.branch_angle
        param.N_it = self.N_it
        param.save = False

        logger.debug(f"Built Parameters for {mesh_suffix}: {param}")
        return param

    def _initialize(self):
        LV1, LV2, RV1, RV2 = self.meshes_list

        logger.info("Building LV fractal tree...")
        param_LV = self._build_fractal_tree_parameters(MeshSuffix.LV, (LV1, LV2))
        LVfractaltree = FractalTree(param_LV)

        logger.info("Building RV fractal tree...")
        param_RV = self._build_fractal_tree_parameters(MeshSuffix.RV, (RV1, RV2))
        RVfractaltree = FractalTree(param_RV)

        return LVfractaltree, RVfractaltree

    def _apply_modifications_to_tree(
        self,
        LVtree_obj: FractalTree,
        RVtree_obj: FractalTree,
        param_dict: dict[str, Any],
        side: str,
    ) -> None:
        """
        Apply parameter modifications to LV/RV fractal trees.

        Args:
            LVtree_obj: FractalTree instance for LV.
            RVtree_obj: FractalTree instance for RV.
            param_dict: Dictionary of parameter name → value.
            side: 'LV', 'RV' or 'both'.
        """

        def _set_param(tree_obj, tree_name: str, param_key: str, param_val: Any):
            if hasattr(tree_obj.params, param_key):
                setattr(tree_obj.params, param_key, param_val)
            else:
                logger.warning(
                    f"Parameter '{param_key}' not found in {tree_name}.params."
                )

        valid_sides = {"LV", "RV", "both"}
        if side not in valid_sides:
            raise ValueError(
                f"Invalid 'side' argument '{side}'. Must be one of {valid_sides}."
            )

        for key, value in param_dict.items():
            if key in ("root_time", "cv"):
                continue

            match side:
                case "LV":
                    _set_param(LVtree_obj, "LVtree", key, value)
                case "RV":
                    _set_param(RVtree_obj, "RVtree", key, value)
                case "both":
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        lv_val, rv_val = value
                        _set_param(LVtree_obj, "LVtree", key, lv_val)
                        _set_param(RVtree_obj, "RVtree", key, rv_val)
                    else:
                        raise ValueError(
                            f"Expected list/tuple of length 2 for '{key}' when side='both'."
                        )

    def _initialize_coupling_points(
        self,
        LVtree: PurkinjeTree,
        RVtree: PurkinjeTree,
        kwargs: dict[str, Any],
        modify: bool,
    ) -> tuple[
        onp.ndarray,  # x0
        onp.ndarray,  # x0_xyz
        onp.ndarray,  # x0_vals
        onp.ndarray,  # SIDE_LV
        onp.ndarray,  # SIDE_RV
        int,          # LVroot
        float,        # LVroot_time
        int,          # RVroot
        float         # RVroot_time
    ]:
        """
        Set up coupling points and parameters like root_time and cv.

        Returns:
            Tuple containing:
            - x0, x0_xyz, x0_vals: activation structures
            - SIDE_LV, SIDE_RV: masks for LV/RV
            - LVroot, LVroot_time, RVroot, RVroot_time
        """

        def _compute_root_times(root_t):
            return 0, -1.0 * onp.min([0.0, root_t]), 0, onp.max([0.0, root_t])

        def _assign_conduction_velocity(tree, kwargs):
            tree.cv = kwargs.get("cv", 2.5)

        def _save_pmj(LVpmj, RVpmj, pat):
            if LVpmj.dtype.kind in {"i"} and RVpmj.dtype.kind in {"i"}:
                onp.savetxt(f"output/patient{pat}/LVpmj.txt", LVpmj, fmt="%d")
                onp.savetxt(f"output/patient{pat}/RVpmj.txt", RVpmj, fmt="%d")
            else:
                raise ValueError("PMJs indices must be integers")

        dir, model = os.path.split(self.patient)
        pat = model.split("-")[0]

        # Default timing and velocity
        LVroot, LVroot_time, RVroot, RVroot_time = 0, 0.0, 0, 0.0
        if modify:
            _assign_conduction_velocity(LVtree, kwargs)
            _assign_conduction_velocity(RVtree, kwargs)
            if "root_time" in kwargs:
                LVroot, LVroot_time, RVroot, RVroot_time = _compute_root_times(
                    kwargs["root_time"]
                )

        # PMJs
        LVpmj = LVtree.pmj
        RVpmj = RVtree.pmj
        LVpmj_vals = onp.full_like(LVpmj, onp.inf, dtype=float)
        RVpmj_vals = onp.full_like(RVpmj, onp.inf, dtype=float)

        if self.save_pmjs:
            _save_pmj(LVpmj, RVpmj, pat)

        PVCs = onp.array([], dtype=int)
        PVCs_vals = onp.array([], dtype=float)

        x0 = onp.r_[LVpmj, RVpmj, PVCs]
        x0_xyz = onp.r_[
            LVtree.xyz[LVpmj, :], RVtree.xyz[RVpmj, :], self.myocardium.xyz[PVCs, :]
        ]
        x0_side = onp.r_[
            onp.repeat("L", LVpmj.size),
            onp.repeat("R", RVpmj.size),
            onp.repeat("M", PVCs.size),
        ]
        x0_vals = onp.r_[LVpmj_vals, RVpmj_vals, PVCs_vals]

        SIDE_LV = x0_side == "L"
        SIDE_RV = x0_side == "R"

        return (
            x0,
            x0_xyz,
            x0_vals,
            SIDE_LV,
            SIDE_RV,
            LVroot,
            LVroot_time,
            RVroot,
            RVroot_time,
        )

    def _activate_purkinje_and_myo(
        self,
        LVtree: PurkinjeTree,
        RVtree: PurkinjeTree,
        x0: onp.ndarray,
        x0_xyz: onp.ndarray,
        x0_vals: onp.ndarray,
        LVroot: int,
        LVroot_time: float,
        RVroot: int,
        RVroot_time: float,
        SIDE_LV: onp.ndarray,
        SIDE_RV: onp.ndarray,
    ) -> onp.ndarray:
        # Activar árboles de Purkinje
        x0_vals[SIDE_LV] = LVtree.activate_fim(
            onp.r_[LVroot, x0[SIDE_LV]], onp.r_[LVroot_time, x0_vals[SIDE_LV]]
        )
        x0_vals[SIDE_RV] = RVtree.activate_fim(
            onp.r_[RVroot, x0[SIDE_RV]], onp.r_[RVroot_time, x0_vals[SIDE_RV]]
        )

        # Activar el miocardio
        myo_vals = self.myocardium.activate_fim(x0_xyz, x0_vals, return_only_pmjs=True)

        # Número de sitios activados temprano
        nnLV = (myo_vals[SIDE_LV] - x0_vals[SIDE_LV] < 0).sum()
        nnRV = (myo_vals[SIDE_RV] - x0_vals[SIDE_RV] < 0).sum()

        logger.info(f"Activation step: nLV = {nnLV}, nRV = {nnRV}")

        # Actualización de tiempos: tomar mínimo
        x0_vals[SIDE_LV] = onp.minimum(x0_vals[SIDE_LV], myo_vals[SIDE_LV])
        x0_vals[SIDE_RV] = onp.minimum(x0_vals[SIDE_RV], myo_vals[SIDE_RV])

        return x0_vals

    def run_ECG(
        self,
        LVfractaltree: FractalTree = None,
        RVfractaltree: FractalTree = None,
        n_sim: int = 0,
        modify: bool = False,
        side: str = "both",
        **kwargs: Any,
    ) -> tuple[onp.ndarray, PurkinjeTree, PurkinjeTree]:

        LVfractaltree = self.LVfractaltree if LVfractaltree is None else LVfractaltree
        RVfractaltree = self.RVfractaltree if RVfractaltree is None else RVfractaltree

        if modify:
            self._apply_modifications_to_tree(
                LVfractaltree, RVfractaltree, kwargs, side
            )

        LVfractaltree.grow_tree()
        LVtree = PurkinjeTree(
            onp.array(LVfractaltree.nodes_xyz),
            onp.array(LVfractaltree.connectivity),
            onp.array(LVfractaltree.end_nodes),
        )

        RVfractaltree.grow_tree()
        RVtree = PurkinjeTree(
            onp.array(RVfractaltree.nodes_xyz),
            onp.array(RVfractaltree.connectivity),
            onp.array(RVfractaltree.end_nodes),
        )

        (
            x0,
            x0_xyz,
            x0_vals,
            SIDE_LV,
            SIDE_RV,
            LVroot,
            LVroot_time,
            RVroot,
            RVroot_time,
        ) = self._initialize_coupling_points(
            LVtree, RVtree, kwargs=kwargs, modify=modify
        )

        ecg = None

        for k in range(self.kmax):
            logger.info(f"Iteration {k + 1} / {self.kmax}")

            x0_vals = self._activate_purkinje_and_myo(
                LVtree,
                RVtree,
                x0,
                x0_xyz,
                x0_vals,
                LVroot,
                LVroot_time,
                RVroot,
                RVroot_time,
                SIDE_LV,
                SIDE_RV,
            )

            ecg_new = self.myocardium.new_get_ecg(record_array=False).copy()

            if ecg is not None:
                ecg_err = onp.linalg.norm(ecg - ecg_new) / onp.linalg.norm(ecg)
                logger.info(f"ECG error = {ecg_err:.6f}")
                if ecg_err < 1e-2:
                    logger.info("ECG has converged below tolerance threshold.")
                    break

            ecg = ecg_new

        ecg = self.myocardium.new_get_ecg()
        return ecg, LVtree, RVtree

    def save_fractaltrees(self, lv_path: str, rv_path: str) -> bool:
        """
        Save the LV and RV fractal trees to the specified file paths.

        Args:
            lv_path (str): Path to save the LV fractal tree.
            rv_path (str): Path to save the RV fractal tree.

        Returns:
            bool: True if both trees were saved successfully, False otherwise.
        """
        try:
            self.LVfractaltree.save(lv_path)
            self.RVfractaltree.save(rv_path)
            logger.info(f"Saved LV tree to {lv_path} and RV tree to {rv_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save fractal trees: {e}")
            return False
