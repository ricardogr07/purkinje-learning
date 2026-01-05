from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as onp
import numpy as np
import matplotlib.pyplot as plt

# PyVista is used for mesh + leadfield projection
import pyvista as pv

# Purkinje-learning classes (you said: use these)
from purkinje_learning.bo_purkinje_tree import BO_PurkinjeTree, BO_PurkinjeTreeConfig
from purkinje_learning.bo_ecg import BO_ecg, OptimParam

# Myocardial mesh (new implementation)
from myocardial_mesh import MyocardialMesh

# JAXBO (installed via your JAX-BO package / deps)
try:
    from jaxbo.models import GP  # noqa: F401
    from jaxbo.utils import normalize  # noqa: F401
    JAXBO_AVAILABLE = True
except Exception:
    JAXBO_AVAILABLE = False


# -------------------------
# Defaults for YOUR machine
# -------------------------
DEFAULT_BASE = Path(r"C:\Users\ricar\Downloads\karli\karli")
DEFAULT_PATIENT_PREFIX = DEFAULT_BASE / "S62_BP_structs_2lyr"

DEFAULT_MESH_VTK = DEFAULT_BASE / "S62_BP_structs_2lyr_mesh_oriented.vtk"
DEFAULT_FIBERS_VTK = DEFAULT_BASE / "S62_BP_structs_2lyr_f0_oriented.vtk"

DEFAULT_LEADF_DIR = DEFAULT_BASE / "leadfields"
DEFAULT_MAP_CSV = DEFAULT_BASE / "S62_BP_structs_2lyr.torso_ecg_locs.header.nodes.csv"

DEFAULT_OBS_JSON = DEFAULT_BASE / (
    "ecg12lead.scale1.0.t1.0.refecg_S62.v2.xls.json.filt_o3_n2000.0_l150.0_h0.5_bs48.0_52.0.json.mean_beat.json"
)

DEFAULT_OUT_ROOT = DEFAULT_BASE / "poc_17d_bo_lite_runs"


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# -------------------------
# Logging
# -------------------------
def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("poc_17d_bo_lite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# -------------------------
# JSON helpers
# -------------------------
def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    # numpy / jax arrays
    if hasattr(x, "tolist"):
        return x.tolist()
    # fallback
    return str(x)


# -------------------------
# Leadfields: load volumetric, project to surface
# -------------------------
def load_leadfields_volumetric(leadfields_dir: Path, mapping_csv: Path) -> Dict[str, onp.ndarray]:
    import pandas as pd

    df = pd.read_csv(mapping_csv, sep=r"\s+")
    df["elec"] = df["elec"].astype(str).str.strip()
    df = df[df["elec"] != "RL"]  # reference electrode -> no .dat

    lf_vol: Dict[str, onp.ndarray] = {}
    for _, row in df.iterrows():
        elec = row["elec"]
        node = int(row["node"])
        f = leadfields_dir / f"LF_Z_extra_Ref_347195_Field_{node}.dat"
        lf_vol[elec] = onp.loadtxt(f, dtype=onp.float32)

    return lf_vol


def project_leadfields_to_surface(mesh_vtk: Path, lf_vol: dict[str, np.ndarray], patient_prefix: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Project volumetric leadfields (size ~ Nvol) to surface nodes (Nsurf)
    using vol_id mapping.

    Priority:
      1) mesh.point_data["vol_id"] if present
      2) {patient_prefix}.biv.nod (binary int32 pairs; vol_id is first int of each pair)
    """
    mesh = pv.read(str(mesh_vtk))

    vol_id = None
    if "vol_id" in mesh.point_data:
        vol_id = np.asarray(mesh.point_data["vol_id"]).astype(np.int64)
    else:
        nod = patient_prefix.with_suffix(".biv.nod")
        if not nod.exists():
            raise RuntimeError(
                "Mesh is missing point_data['vol_id'] and .biv.nod was not found.\n"
                f"Expected: {nod}\n"
                "Fix: pass correct --patient-prefix or provide a mesh that includes vol_id."
            )

        arr = np.fromfile(nod, dtype=np.int32)
        if arr.size % 2 != 0:
            raise RuntimeError(f"Unexpected .biv.nod size (int32 count not even): {arr.size} in {nod}")

        vol_id = arr[0::2].astype(np.int64)  # first of each pair
        if vol_id.size != mesh.n_points:
            # Not fatal, but it's a red flag: ordering mismatch between mesh points and biv.nod mapping.
            raise RuntimeError(
                f"vol_id length mismatch: biv.nod has {vol_id.size} ids but mesh has {mesh.n_points} points.\n"
                f"mesh_vtk={mesh_vtk}\n"
                f"biv_nod={nod}\n"
                "Fix: use the matching surface mesh for this patient (same ordering as biv.nod)."
            )

    # detect 1-based vs 0-based once, using any leadfield size
    any_v = next(iter(lf_vol.values()))
    if vol_id.min() >= 1 and vol_id.max() <= any_v.size and (vol_id.max() == any_v.size or vol_id.min() == 1):
        vol_idx = vol_id - 1
    else:
        vol_idx = vol_id

    lf_surf: dict[str, np.ndarray] = {}
    for k, v in lf_vol.items():
        if vol_idx.min() < 0 or vol_idx.max() >= v.size:
            raise RuntimeError(
                f"vol_id out of bounds for leadfield {k}: "
                f"vol_idx in [{vol_idx.min()},{vol_idx.max()}], v.size={v.size}"
            )
        lf_surf[k] = v[vol_idx].astype(np.float32, copy=False)

    return lf_surf, vol_idx


# -------------------------
# Observed ECG loader (robust-ish for your JSON)
# -------------------------
def load_obs_ecg_json(path: Path) -> Tuple[Dict[str, onp.ndarray], float]:
    d = json.loads(path.read_text(encoding="utf-8"))

    # Heuristic: find a dict that contains the 12 leads
    def find_lead_dict(obj: Any) -> Dict[str, Any] | None:
        if isinstance(obj, dict):
            keys = set(obj.keys())
            if all(k in keys for k in LEADS_12):
                return obj
            for v in obj.values():
                r = find_lead_dict(v)
                if r is not None:
                    return r
        if isinstance(obj, list):
            for it in obj:
                r = find_lead_dict(it)
                if r is not None:
                    return r
        return None

    lead_dict = find_lead_dict(d)
    if lead_dict is None:
        raise RuntimeError(f"Could not find 12-lead signals inside JSON: {path}")

    # dt might be present, otherwise estimate from common defaults
    dt = None
    if isinstance(d, dict):
        for cand in ["dt", "sampling_dt", "Ts", "t_step"]:
            if cand in d and isinstance(d[cand], (int, float)):
                dt = float(d[cand])
                break
    # fallback: your mean beat tends to be 0.5 ms
    if dt is None:
        dt = 0.0005

    obs = {k: onp.asarray(lead_dict[k], dtype=onp.float64).ravel() for k in LEADS_12}
    n = min(len(obs[k]) for k in LEADS_12)
    obs = {k: obs[k][:n] for k in LEADS_12}

    return obs, dt


# -------------------------
# Sim ECG to dict (from structured array or dict)
# -------------------------
def sim_to_lead_dict(ecg_sim):
    """
    Convert the simulated ECG (structured array or dict-like) to a dict with
    standard 12-lead names: I, II, III, aVR, aVL, aVF, V1..V6.

    Accepts aliases from MyocardialMesh:
      E1->I, E2->II, E3->III
    """
    import numpy as np

    required = ["I", "II", "III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6"]

    alias = {
        "E1": "I",
        "E2": "II",
        "E3": "III",
    }

    # ecg_sim can be a structured array (dtype.names) or a dict-like
    if hasattr(ecg_sim, "dtype") and getattr(ecg_sim.dtype, "names", None):
        names = list(ecg_sim.dtype.names)

        out = {}
        # 1) copy direct matches
        for k in names:
            if k in required:
                out[k] = np.asarray(ecg_sim[k], dtype=np.float64)

        # 2) apply aliases (E1/E2/E3)
        for src, dst in alias.items():
            if dst not in out and src in names:
                out[dst] = np.asarray(ecg_sim[src], dtype=np.float64)

        missing = [k for k in required if k not in out]
        if missing:
            raise RuntimeError(
                "Sim ECG is missing required standard leads.\n"
                f"Missing: {missing}\n"
                f"Found: {names}\n"
                "Fix: ensure sim outputs {I,II,III,...} or aliases {E1,E2,E3}."
            )
        return out

    # dict-like case
    if isinstance(ecg_sim, dict):
        names = list(ecg_sim.keys())
        out = {}

        for k in names:
            if k in required:
                out[k] = np.asarray(ecg_sim[k], dtype=np.float64)

        for src, dst in alias.items():
            if dst not in out and src in ecg_sim:
                out[dst] = np.asarray(ecg_sim[src], dtype=np.float64)

        missing = [k for k in required if k not in out]
        if missing:
            raise RuntimeError(
                "Sim ECG dict is missing required standard leads.\n"
                f"Missing: {missing}\n"
                f"Found: {names}\n"
                "Fix: ensure sim outputs {I,II,III,...} or aliases {E1,E2,E3}."
            )
        return out

    raise TypeError(f"Unsupported sim ECG type: {type(ecg_sim)}")


# -------------------------
# QRS window + LSQ scaling metrics
# -------------------------
def find_r_peak_index(x: onp.ndarray) -> int:
    # Robust enough for mean beat: max abs in lead II works fine for your data
    return int(onp.argmax(onp.abs(x)))


def apply_shift_pad(x: onp.ndarray, shift: int) -> onp.ndarray:
    """
    shift > 0: move right (delay) -> pad left with zeros
    shift < 0: move left (advance) -> pad right with zeros
    """
    n = x.size
    y = onp.zeros_like(x)
    if shift == 0:
        return x.copy()

    if shift > 0:
        if shift < n:
            y[shift:] = x[: n - shift]
    else:
        s = -shift
        if s < n:
            y[: n - s] = x[s:]
    return y


def best_shift_by_xcorr(a: onp.ndarray, b: onp.ndarray, max_shift: int = 200) -> int:
    """
    Return shift applied to b to best match a (based on cross-correlation).
    """
    # limit to a window around the peak for stability
    cc = onp.correlate(a, b, mode="full")
    shift = int(onp.argmax(cc) - (b.size - 1))
    shift = max(-max_shift, min(max_shift, shift))
    return shift


def qrs_slices(
    n: int,
    dt: float,
    r_idx: int,
    qrs_pre: float,
    qrs_post: float,
    baseline0: float,
    baseline1: float,
) -> Tuple[slice, slice]:
    def t_to_i(t: float) -> int:
        return int(round(t / dt))

    i0 = max(0, r_idx - t_to_i(qrs_pre))
    i1 = min(n, r_idx + t_to_i(qrs_post))

    b0 = max(0, r_idx + t_to_i(baseline0))
    b1 = max(0, min(n, r_idx + t_to_i(baseline1)))

    return slice(i0, i1), slice(b0, b1)


def metrics_qrs_lsq(
    obs: Dict[str, onp.ndarray],
    sim: Dict[str, onp.ndarray],
    dt_obs: float,
    dt_sim: float,
    qrs_pre: float = 0.04,
    qrs_post: float = 0.12,
    baseline0: float = -0.04,
    baseline1: float = -0.02,
    normalize: str = "rms",
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Returns:
      mean_rmse_norm, mean_rho, per-lead rows
    """
    # Resample sim to obs length/time-grid if needed (simple linear interpolation)
    n_obs = obs["II"].size
    n_sim = sim["II"].size

    if dt_sim <= 0:
        dt_sim = dt_obs

    if n_sim != n_obs or abs(dt_sim - dt_obs) > 1e-9:
        t_obs = onp.arange(n_obs) * dt_obs
        t_sim = onp.arange(n_sim) * dt_sim
        sim_rs = {}
        for k in LEADS_12:
            sim_rs[k] = onp.interp(t_obs, t_sim, sim[k]).astype(onp.float64)
        sim = sim_rs

    # Align by cross-correlation on lead II
    r_obs = find_r_peak_index(obs["II"])
    r_sim = find_r_peak_index(sim["II"])
    # First coarse align by peak, then refine by xcorr around QRS-ish
    coarse_shift = r_obs - r_sim
    sim_coarse = {k: apply_shift_pad(sim[k], coarse_shift) for k in LEADS_12}
    refine_shift = best_shift_by_xcorr(obs["II"], sim_coarse["II"], max_shift=250)
    total_shift = coarse_shift + refine_shift
    sim_aligned = {k: apply_shift_pad(sim[k], total_shift) for k in LEADS_12}

    qrs_sl, base_sl = qrs_slices(
        n=n_obs,
        dt=dt_obs,
        r_idx=r_obs,
        qrs_pre=qrs_pre,
        qrs_post=qrs_post,
        baseline0=baseline0,
        baseline1=baseline1,
    )

    rows: List[Dict[str, float]] = []
    rmse_norms = []
    rhos = []

    eps = 1e-12
    for k in LEADS_12:
        y = obs[k].copy()
        x = sim_aligned[k].copy()

        # Baseline correction using observed baseline window
        if base_sl.stop > base_sl.start:
            y0 = float(onp.mean(y[base_sl]))
            x0 = float(onp.mean(x[base_sl]))
            y = y - y0
            x = x - x0

        yq = y[qrs_sl]
        xq = x[qrs_sl]

        denom = float(onp.dot(xq, xq)) + eps
        alpha = float(onp.dot(yq, xq) / denom)
        xq_fit = alpha * xq

        rmse_raw = float(onp.sqrt(onp.mean((yq - xq_fit) ** 2)))
        if normalize == "rms":
            rms_obs = float(onp.sqrt(onp.mean(yq**2))) + eps
            rmse_norm = rmse_raw / rms_obs
        else:
            rmse_norm = rmse_raw

        # Pearson rho
        yq0 = yq - float(onp.mean(yq))
        xq0 = xq_fit - float(onp.mean(xq_fit))
        num = float(onp.dot(yq0, xq0))
        den = float(onp.sqrt(onp.dot(yq0, yq0) * onp.dot(xq0, xq0))) + eps
        rho = num / den

        rows.append(
            dict(
                lead=k,
                alpha=alpha,
                rmse_raw=rmse_raw,
                rmse_norm=rmse_norm,
                rho=rho,
                abs_rho=abs(rho),
            )
        )
        rmse_norms.append(rmse_norm)
        rhos.append(rho)

    mean_rmse_norm = float(onp.mean(rmse_norms))
    mean_rho = float(onp.mean(rhos))
    return mean_rmse_norm, mean_rho, rows


# -------------------------
# Save artifacts
# -------------------------
def save_npz_from_leads(path: Path, leads: Dict[str, onp.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onp.savez(path, **{k: onp.asarray(leads[k]) for k in LEADS_12})


def save_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lead", "alpha", "rmse_raw", "rmse_norm", "rho", "abs_rho"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_overlay(out_png, obs, sim, title=""):
    """
    Plot stacked 12-lead overlay for observed vs simulated signals.

    Handles different lengths (e.g., obs 1400 samples vs sim 206 samples)
    by resampling simulated to observed length via linear interpolation.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    # Determine reference length from obs (assume all obs leads same length)
    n_obs = len(obs[leads[0]])
    t = np.arange(n_obs, dtype=np.float64)

    def _to_len(x, n_target):
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        if n == n_target:
            return x
        if n < 2:
            return np.full(n_target, float(x[0]) if n == 1 else 0.0, dtype=np.float64)
        # resample by interpolation (index domain)
        xp = np.linspace(0.0, 1.0, n, dtype=np.float64)
        xq = np.linspace(0.0, 1.0, n_target, dtype=np.float64)
        return np.interp(xq, xp, x)

    plt.figure(figsize=(12, 10))
    offset = 0.0
    # Auto offset based on obs range for readability
    # (robust if some leads are tiny)
    per_lead_scale = []
    for ld in leads:
        xo = np.asarray(obs[ld], dtype=np.float64).ravel()
        per_lead_scale.append(np.nanmax(xo) - np.nanmin(xo))
    dy = np.nanmedian([s for s in per_lead_scale if np.isfinite(s) and s > 0])
    if not np.isfinite(dy) or dy <= 0:
        dy = 1.0
    dy *= 1.6  # spacing factor

    for i, ld in enumerate(leads):
        xo = np.asarray(obs[ld], dtype=np.float64).ravel()
        xs = _to_len(sim[ld], n_obs)

        # baseline shift per lead
        y0 = i * dy

        # obs solid, sim dashed
        plt.plot(t, xo + y0, linewidth=1.2)
        plt.plot(t, xs + y0, linewidth=1.0, linestyle="--")

        # label at left
        plt.text(t[0], y0, ld, va="bottom", ha="left", fontsize=9)

    plt.title(title)
    plt.xlabel("sample index (aligned to obs length)")
    plt.yticks([])
    plt.tight_layout()
    out_png = str(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_top3(path: Path, obs: Dict[str, onp.ndarray], sim: Dict[str, onp.ndarray], top3: List[str], title: str) -> None:
    fig = plt.figure(figsize=(14, 6))
    n = obs["II"].size
    t = onp.arange(n)

    for k in top3:
        plt.plot(t, obs[k], linewidth=1.2, label=f"{k} obs")
        plt.plot(t, sim[k], linewidth=1.2, linestyle="--", label=f"{k} sim")

    plt.title(title)
    plt.xlabel("sample")
    plt.ylabel("signal (raw units)")
    plt.legend(ncol=3)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -------------------------
# 17D param spec + mapping via BO_ecg helper
# -------------------------
def build_var_params_17d() -> List[OptimParam]:
    """
    17D layout (matches what we've been running):
      init_length: 2
      length: 1
      w: 1
      l_segment: 2
      fascicles_length: 4
      fascicles_angles: 4
      branch_angle: 1
      root_time: 1
      cv: 1
    """
    vp: List[OptimParam] = []

    # NOTE: your BO_ecg.set_dictionary_variables duplicates some scalars into 2-element lists;
    #       we keep the exact shapes used in your PoC.
    vp.append(OptimParam("init_length", lower=onp.array([1.0, 1.0]), upper=onp.array([5.0, 5.0]), prior="uniform"))
    vp.append(OptimParam("length", lower=onp.array([4.0]), upper=onp.array([12.0]), prior="uniform"))
    vp.append(OptimParam("w", lower=onp.array([0.0]), upper=onp.array([1.0]), prior="uniform"))
    vp.append(OptimParam("l_segment", lower=onp.array([0.1, 0.1]), upper=onp.array([3.0, 3.0]), prior="uniform"))
    vp.append(OptimParam("fascicles_length", lower=onp.array([2.0, 2.0, 2.0, 2.0]), upper=onp.array([10.0, 10.0, 10.0, 10.0]), prior="uniform"))
    vp.append(OptimParam("fascicles_angles", lower=onp.array([0.1, 0.1, 0.1, 0.1]), upper=onp.array([1.2, 1.2, 1.2, 1.2]), prior="uniform"))
    vp.append(OptimParam("branch_angle", lower=onp.array([0.1]), upper=onp.array([1.3]), prior="uniform"))
    vp.append(OptimParam("root_time", lower=onp.array([-75.0]), upper=onp.array([50.0]), prior="uniform"))
    vp.append(OptimParam("cv", lower=onp.array([2.0]), upper=onp.array([4.0]), prior="uniform"))

    return vp


def flatten_bounds(var_params: List[OptimParam]) -> Tuple[onp.ndarray, onp.ndarray]:
    lb = onp.concatenate([onp.asarray(p.lower).ravel() for p in var_params]).astype(onp.float64)
    ub = onp.concatenate([onp.asarray(p.upper).ravel() for p in var_params]).astype(onp.float64)
    if lb.shape != ub.shape:
        raise RuntimeError("Bounds shape mismatch.")
    return lb, ub


def midpoint(lb: onp.ndarray, ub: onp.ndarray) -> onp.ndarray:
    return 0.5 * (lb + ub)


def random_perturb(x0: onp.ndarray, lb: onp.ndarray, ub: onp.ndarray, frac: float, rng: onp.random.Generator) -> onp.ndarray:
    """
    Multiplicative-ish perturbation around x0, clipped to bounds.
    For parameters that can be negative (root_time), use additive range.
    """
    x = x0.copy()
    for i in range(x.size):
        lo, hi = lb[i], ub[i]
        span = hi - lo
        if span <= 0:
            continue

        # If range crosses 0 or is clearly "time-like", do additive perturbation
        if lo < 0 < hi:
            delta = (rng.uniform(-1.0, 1.0) * frac) * span
            x[i] = x[i] + delta
        else:
            # multiplicative around x0, with additive fallback
            mult = 1.0 + rng.uniform(-frac, frac)
            cand = x[i] * mult
            # if x0 is near 0, multiplicative is useless
            if abs(x[i]) < 1e-9:
                cand = x[i] + (rng.uniform(-1.0, 1.0) * frac) * span
            x[i] = cand

        x[i] = min(hi, max(lo, x[i]))

    return x


# -------------------------
# Core evaluation: run forward + compute metrics
# -------------------------
def evaluate_point(
    tag: str,
    x: onp.ndarray,
    out_dir: Path,
    var_params: List[OptimParam],
    bo_ecg: BO_ecg,
    bo_tree: BO_PurkinjeTree,
    obs: Dict[str, onp.ndarray],
    dt_obs: float,
    dt_sim: float,
    logger: logging.Logger,
) -> Dict[str, Any]:
    t0 = time.time()
    eval_dir = out_dir / tag
    eval_dir.mkdir(parents=True, exist_ok=True)

    # map x -> dict params via BO_ecg helper (keeps same semantics as your demo code)
    dict_params = bo_ecg.set_dictionary_variables(var_params, x)

    # Save params
    (eval_dir / "params.json").write_text(json.dumps(_jsonable(dict_params), indent=2), encoding="utf-8")

    # Run expensive forward
    logger.info("[%s] running forward (expensive) ...", tag)
    ecg_sim, _, _ = bo_tree.run_ECG(modify=True, side="both", **dict_params)

    # Convert to lead dict
    sim = sim_to_lead_dict(ecg_sim)
    save_npz_from_leads(eval_dir / "sim_ecg.npz", sim)

    # Compute metrics (QRS + LSQ scaling)
    mean_rmse_norm, mean_rho, rows = metrics_qrs_lsq(
        obs=obs, sim=sim, dt_obs=dt_obs, dt_sim=dt_sim,
        qrs_pre=0.04, qrs_post=0.12, baseline0=-0.04, baseline1=-0.02, normalize="rms",
    )
    save_metrics_csv(eval_dir / "metrics_qrs_lsq.csv", rows)

    # top3 by |rho|
    rows_sorted = sorted(rows, key=lambda r: r["abs_rho"], reverse=True)
    top3 = [rows_sorted[i]["lead"] for i in range(min(3, len(rows_sorted)))]
    with open(eval_dir / "top3_by_abs_rho.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lead", "alpha", "rmse_raw", "rmse_norm", "rho", "abs_rho"])
        w.writeheader()
        for r in rows_sorted[:3]:
            w.writerow(r)

    # figures
    #plot_overlay(eval_dir / "overlay_obs_vs_sim_lsq.png", obs, sim, title=f"{tag} | mean RMSE(norm)={mean_rmse_norm:.4f} | mean rho={mean_rho:.4f}")
    #plot_top3(eval_dir / "top3_by_abs_rho_lsq.png", obs, sim, top3=top3, title=f"{tag} | top3 by |rho|: {', '.join(top3)}")

    t1 = time.time()
    return dict(
        tag=tag,
        mean_rmse_norm=float(mean_rmse_norm),
        mean_rho=float(mean_rho),
        time_forward_sec=float(t1 - t0),
        top3=top3,
    )


# -------------------------
# GP/EI step (lite)
# -------------------------
def gp_ei_suggest_next(
    X: onp.ndarray,
    y: onp.ndarray,
    lb: onp.ndarray,
    ub: onp.ndarray,
    seed: int,
    n_candidates: int = 2000,
) -> onp.ndarray:
    """
    Very small "lite" proposal:
      - sample a candidate pool uniformly in bounds
      - fit GP via JAXBO
      - score EI on candidates
      - return best candidate

    If JAXBO is unavailable or anything fails, raises.
    """
    if not JAXBO_AVAILABLE:
        raise RuntimeError("JAXBO not available")

    # Local imports keep the top clean
    import jax.numpy as jnp
    from jaxbo.models import GP
    from jaxbo.utils import normalize
    from jaxbo.acquisitions import EI

    rng = onp.random.default_rng(seed)

    dim = X.shape[1]
    Xcand = rng.uniform(lb, ub, size=(n_candidates, dim)).astype(onp.float64)

    # JAX arrays
    X_j = jnp.asarray(X, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    Xc_j = jnp.asarray(Xcand, dtype=jnp.float64)

    bounds = jnp.stack([jnp.asarray(lb), jnp.asarray(ub)], axis=0)

    # normalize as JAXBO expects
    Xn, yn = normalize(X_j, y_j, bounds)

    # minimal GP options (keep it simple, ARD on)
    options = dict(
        dim=dim,
        ARD=True,
        # kernel name may differ across versions; keep default if possible
    )

    gp = GP(options)
    gp.fit(Xn, yn)

    # predict on candidates
    mu, var = gp.predict(Xc_j, Xn, yn)
    sigma = jnp.sqrt(jnp.maximum(var, 1e-12))

    # EI: minimization -> use best (min) yn so far
    fbest = jnp.min(yn)
    ei = EI(mu, sigma, fbest)

    idx = int(jnp.argmax(ei))
    return onp.asarray(Xcand[idx], dtype=onp.float64)


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--patient-prefix", type=Path, default=DEFAULT_PATIENT_PREFIX)

    p.add_argument("--mesh-vtk", type=Path, default=DEFAULT_MESH_VTK)
    p.add_argument("--fibers-vtk", type=Path, default=DEFAULT_FIBERS_VTK)

    p.add_argument("--leadfields-dir", type=Path, default=DEFAULT_LEADF_DIR)
    p.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAP_CSV)

    p.add_argument("--obs-json", type=Path, default=DEFAULT_OBS_JSON)

    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)

    # Seeds for LV/RV (meshes_list in BO_PurkinjeTree)
    p.add_argument("--lv-roots", type=int, nargs="+", default=[742, 984])
    p.add_argument("--rv-roots", type=int, nargs="+", default=[282, 195])

    # Loop config
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--n-evals", type=int, default=10, help="Number of post-baseline evaluations (baseline + n_evals total).")
    p.add_argument("--n-random", type=int, default=3, help="How many of the post-baseline evaluations are random perturbations.")
    p.add_argument("--perturb-frac", type=float, default=0.20)

    # Sim dt (your forward typically uses 1 ms)
    p.add_argument("--dt-sim", type=float, default=0.001)

    # MyocardialMesh device
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    logger.info("Output: %s", run_dir)

    # Save config snapshot
    (run_dir / "config.json").write_text(json.dumps(_jsonable(vars(args)), indent=2), encoding="utf-8")

    # Leadfields: load volumetric then project to surface
    t0 = time.time()
    lf_vol = load_leadfields_volumetric(args.leadfields_dir, args.mapping_csv)
    lf_surf, vol_id = project_leadfields_to_surface(args.mesh_vtk, lf_vol, args.patient_prefix)
    t1 = time.time()
    logger.info("Leadfields projected: n_surf=%d in %.3fs", vol_id.size, t1 - t0)
    logger.info("Leadfield keys: %s", list(lf_surf.keys()))
    logger.info("Leadfield (projected) sizes: %s", {k: int(v.size) for k, v in lf_surf.items()})

    # Observed ECG
    obs, dt_obs = load_obs_ecg_json(args.obs_json)
    logger.info("Obs ECG loaded: n=%d dt~%.6fs", obs["II"].size, dt_obs)

    # MyocardialMesh init (needs your patch: electrodes_position=None + lead_fields_dict)
    myo = MyocardialMesh(
        mesh_path=str(args.mesh_vtk),
        electrodes_position=None,
        fibers_path=str(args.fibers_vtk),
        device=args.device,
        conductivity_params=None,
        lead_fields_dict=lf_surf,
    )
    n_nodes = onp.asarray(myo.xyz).shape[0]
    logger.info("Myocardium mesh points: %d", n_nodes)

    # BO_PurkinjeTree config
    meshes_list = list(args.lv_roots) + list(args.rv_roots)  # [LV1, LV2, RV1, RV2]
    cfg = BO_PurkinjeTreeConfig(
        # Reasonable defaults; actual values will be overridden by dict_params in run_ECG(modify=True)
        init_length=[3.0, 3.0],
        length=[8.0, 8.0],
        w=[0.5, 0.5],
        l_segment=[1.55, 1.55],
        fascicles_length=[6.0, 6.0, 6.0, 6.0],
        fascicles_angles=[0.65, 0.65, 0.65, 0.65],
        branch_angle=[0.7, 0.7],
        N_it=10,
    )

    bo_tree = BO_PurkinjeTree(str(args.patient_prefix), meshes_list, cfg, myo)

    # BO_ecg instance only used for param mapping helper
    bo_ecg = BO_ecg(bo_tree)

    # 17D params + bounds
    var_params = build_var_params_17d()
    lb, ub = flatten_bounds(var_params)
    dim = lb.size
    x0 = midpoint(lb, ub)

    # History holders
    history_rows: List[Dict[str, Any]] = []
    X_list: List[onp.ndarray] = []
    y_list: List[float] = []

    rng = onp.random.default_rng(args.seed)

    # ---- Baseline
    baseline = evaluate_point(
        tag="baseline_x0_midpoint",
        x=x0,
        out_dir=run_dir,
        var_params=var_params,
        bo_ecg=bo_ecg,
        bo_tree=bo_tree,
        obs=obs,
        dt_obs=dt_obs,
        dt_sim=float(args.dt_sim),
        logger=logger,
    )
    history_rows.append(baseline)
    X_list.append(x0.copy())
    y_list.append(float(baseline["mean_rmse_norm"]))

    best = baseline
    best_y = float(baseline["mean_rmse_norm"])

    # ---- Post-baseline loop
    n_evals = int(args.n_evals)
    n_random = min(int(args.n_random), n_evals)

    for i in range(n_evals):
        tag = f"eval_{i+1:02d}"
        use_random = i < n_random

        if use_random:
            x_next = random_perturb(x0, lb, ub, frac=float(args.perturb_frac), rng=rng)
            tag2 = f"{tag}_random"
        else:
            tag2 = f"{tag}_gp_ei"
            try:
                # Suggest next point using GP/EI (lite)
                X = onp.stack(X_list, axis=0)
                y = onp.asarray(y_list, dtype=onp.float64)
                x_next = gp_ei_suggest_next(
                    X=X,
                    y=y,
                    lb=lb,
                    ub=ub,
                    seed=args.seed + 1000 + i,
                    n_candidates=2000,
                )
            except Exception as e:
                logger.warning("[%s] GP/EI failed (%r). Falling back to random perturbation.", tag2, e)
                x_next = random_perturb(x0, lb, ub, frac=float(args.perturb_frac), rng=rng)
                tag2 = f"{tag}_fallback_random"

        res = evaluate_point(
            tag=tag2,
            x=x_next,
            out_dir=run_dir,
            var_params=var_params,
            bo_ecg=bo_ecg,
            bo_tree=bo_tree,
            obs=obs,
            dt_obs=dt_obs,
            dt_sim=float(args.dt_sim),
            logger=logger,
        )
        history_rows.append(res)
        X_list.append(x_next.copy())
        y_list.append(float(res["mean_rmse_norm"]))

        # Update best
        if float(res["mean_rmse_norm"]) < best_y:
            best_y = float(res["mean_rmse_norm"])
            best = res

        logger.info(
            "[%s] RMSE(norm)=%.6f rho=%.6f | best_so_far=%.6f (%s)",
            tag2,
            float(res["mean_rmse_norm"]),
            float(res["mean_rho"]),
            best_y,
            best["tag"],
        )

    # ---- Save history.csv
    hist_path = run_dir / "history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "mean_rmse_norm", "mean_rho", "time_forward_sec", "top3"])
        w.writeheader()
        for r in history_rows:
            rr = r.copy()
            rr["top3"] = ",".join(rr.get("top3", []))
            w.writerow(rr)
    logger.info("History: %s", hist_path)

    # ---- Save summary.csv (baseline vs best)
    summ_path = run_dir / "summary.csv"
    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "which",
                "tag",
                "mean_rmse_norm",
                "mean_rho",
                "time_forward_sec",
            ],
        )
        w.writeheader()
        w.writerow(dict(which="baseline", tag=baseline["tag"], mean_rmse_norm=baseline["mean_rmse_norm"], mean_rho=baseline["mean_rho"], time_forward_sec=baseline["time_forward_sec"]))
        w.writerow(dict(which="best", tag=best["tag"], mean_rmse_norm=best["mean_rmse_norm"], mean_rho=best["mean_rho"], time_forward_sec=best["time_forward_sec"]))
    logger.info("Summary: %s", summ_path)

    # ---- Convergence plot
    ys = [float(r["mean_rmse_norm"]) for r in history_rows]
    best_so_far = []
    cur = float("inf")
    for v in ys:
        cur = min(cur, v)
        best_so_far.append(cur)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(onp.arange(len(best_so_far)), best_so_far, linewidth=2.0)
    plt.xlabel("evaluation (0=baseline)")
    plt.ylabel("best-so-far RMSE(norm)")
    plt.title("Convergence (best-so-far)")
    plt.tight_layout()
    conv_path = run_dir / "convergence_best_so_far.png"
    fig.savefig(conv_path, dpi=160)
    plt.close(fig)
    logger.info("Convergence: %s", conv_path)

    logger.info("=== DONE ===")
    logger.info("Best: %s | RMSE(norm)=%.6f rho=%.6f", best["tag"], float(best["mean_rmse_norm"]), float(best["mean_rho"]))


if __name__ == "__main__":
    main()
