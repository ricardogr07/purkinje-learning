from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv

# --- allow running from purkinje-learning/scripts ---
THIS = Path(__file__).resolve()
REPO = THIS.parents[1]  # purkinje-learning/
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from purkinje_learning.bo_purkinje_tree import BO_PurkinjeTree, BO_PurkinjeTreeConfig  # noqa: E402
from purkinje_learning.bo_ecg import BO_ecg, OptimParam  # noqa: E402


# ----------------------------
# Defaults (Karli/S62)
# ----------------------------
DEFAULT_BASE = Path(r"C:\Users\ricar\Downloads\karli\karli")
DEFAULT_PATIENT_PREFIX = DEFAULT_BASE / "S62_BP_structs_2lyr"
DEFAULT_MESH_VTK = DEFAULT_BASE / "S62_BP_structs_2lyr_mesh_oriented.vtk"
DEFAULT_FIBERS_VTK = DEFAULT_BASE / "S62_BP_structs_2lyr_f0_oriented.vtk"
DEFAULT_LEADF_DIR = DEFAULT_BASE / "leadfields"
DEFAULT_MAP_CSV = DEFAULT_BASE / "S62_BP_structs_2lyr.torso_ecg_locs.header.nodes.csv"
DEFAULT_OBS_JSON = DEFAULT_BASE / (
    "ecg12lead.scale1.0.t1.0.refecg_S62.v2.xls.json.filt_o3_n2000.0_l150.0_h0.5_bs48.0_52.0.json.mean_beat.json"
)
DEFAULT_LV_ROOTS = [742, 984]
DEFAULT_RV_ROOTS = [282, 195]


# ----------------------------
# Logging
# ----------------------------
def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("poc17d")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ----------------------------
# Leadfields: load + project volumetric -> surface using vol_id
# ----------------------------
def load_leadfields_volumetric(leadfields_dir: Path, mapping_csv: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(mapping_csv, sep=r"\s+")
    df["elec"] = df["elec"].astype(str).str.strip()
    df = df[df["elec"] != "RL"]  # reference electrode -> no .dat

    lf_vol: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        elec = row["elec"]
        node = int(row["node"])
        f = leadfields_dir / f"LF_Z_extra_Ref_347195_Field_{node}.dat"
        lf_vol[elec] = np.loadtxt(f, dtype=np.float32)
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


# ----------------------------
# Observed ECG JSON
# ----------------------------
def load_obs_json(obs_json: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    obj = json.loads(obs_json.read_text(encoding="utf-8"))
    t = np.asarray(obj["t"], dtype=np.float32)
    ecg = {k: np.asarray(v, dtype=np.float32) for k, v in obj["ecg"].items()}
    return t, ecg


# ----------------------------
# Sim ECG naming normalization
# ----------------------------
def sim_to_12lead_dict(sim: Any) -> dict[str, np.ndarray]:
    if isinstance(sim, dict):
        d = {k: np.asarray(v) for k, v in sim.items()}
    else:
        names = getattr(sim.dtype, "names", None)
        if not names:
            raise RuntimeError("Sim ECG is not dict and not structured array.")
        d = {k: np.asarray(sim[k]) for k in names}

    # E1/E2/E3 -> I/II/III
    if "E1" in d and "I" not in d:
        d["I"] = d.pop("E1")
    if "E2" in d and "II" not in d:
        d["II"] = d.pop("E2")
    if "E3" in d and "III" not in d:
        d["III"] = d.pop("E3")

    needed = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    missing = [k for k in needed if k not in d]
    if missing:
        raise RuntimeError(f"Missing simulated leads: {missing}. Have: {sorted(d.keys())}")
    return {k: np.asarray(d[k], dtype=np.float32) for k in needed}


# ----------------------------
# Metrics (QRS window) with LSQ scaling per lead
# ----------------------------
def qrs_metrics_lsq(
    t_obs: np.ndarray,
    obs: dict[str, np.ndarray],
    t_sim: np.ndarray,
    sim: dict[str, np.ndarray],
    *,
    qrs_pre: float = 0.04,
    qrs_post: float = 0.12,
    baseline0: float = -0.04,
    baseline1: float = -0.02,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, float]]:
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    dt_obs = float(np.median(np.diff(t_obs)))
    dt_sim = float(np.median(np.diff(t_sim)))

    # align by abs peak on lead II
    iobs = int(np.argmax(np.abs(obs["II"])))
    isim = int(np.argmax(np.abs(sim["II"])))

    pre_obs = int(round(qrs_pre / dt_obs))
    post_obs = int(round(qrs_post / dt_obs))
    pre_sim = int(round(qrs_pre / dt_sim))
    post_sim = int(round(qrs_post / dt_sim))

    b0_obs = int(round(baseline0 / dt_obs))
    b1_obs = int(round(baseline1 / dt_obs))
    b0_sim = int(round(baseline0 / dt_sim))
    b1_sim = int(round(baseline1 / dt_sim))

    rows = []
    for ld in leads:
        y_obs = np.asarray(obs[ld], dtype=np.float64)
        y_sim = np.asarray(sim[ld], dtype=np.float64)

        o0, o1 = iobs - pre_obs, iobs + post_obs
        s0, s1 = isim - pre_sim, isim + post_sim
        o0 = max(o0, 0)
        s0 = max(s0, 0)
        o1 = min(o1, y_obs.size)
        s1 = min(s1, y_sim.size)
        L = min(o1 - o0, s1 - s0)
        o1, s1 = o0 + L, s0 + L

        w_obs = y_obs[o0:o1]
        w_sim = y_sim[s0:s1]

        ob0, ob1 = iobs + b0_obs, iobs + b1_obs
        sb0, sb1 = isim + b0_sim, isim + b1_sim
        ob0, ob1 = max(ob0, 0), min(ob1, y_obs.size)
        sb0, sb1 = max(sb0, 0), min(sb1, y_sim.size)
        base_obs = float(np.mean(y_obs[ob0:ob1])) if ob1 > ob0 else 0.0
        base_sim = float(np.mean(y_sim[sb0:sb1])) if sb1 > sb0 else 0.0
        w_obs = w_obs - base_obs
        w_sim = w_sim - base_sim

        denom = float(np.dot(w_sim, w_sim))
        alpha = float(np.dot(w_obs, w_sim) / (denom + eps))
        w_fit = alpha * w_sim

        err = w_obs - w_fit
        rmse_raw = float(np.sqrt(np.mean(err * err)))
        rms_obs = float(np.sqrt(np.mean(w_obs * w_obs)) + eps)
        rmse_norm = float(rmse_raw / rms_obs)

        xo = w_obs - float(np.mean(w_obs))
        xs = w_fit - float(np.mean(w_fit))
        rho = float(np.dot(xo, xs) / (np.sqrt(np.dot(xo, xo) * np.dot(xs, xs)) + eps))

        rows.append(
            dict(
                lead=ld,
                alpha=alpha,
                rmse_raw=rmse_raw,
                rmse_norm=rmse_norm,
                rho=rho,
                abs_rho=abs(rho),
            )
        )

    df = pd.DataFrame(rows).sort_values("lead")
    summary = dict(
        mean_rmse_norm=float(df["rmse_norm"].mean()),
        mean_rho=float(df["rho"].mean()),
        mean_abs_rho=float(df["abs_rho"].mean()),
    )
    return df, summary


# ----------------------------
# 17D variable vector (dim=2)
# ----------------------------
def var_params_17d(dim: int = 2) -> list[OptimParam]:
    """
    17D when dim=2:
      init_length (2)
      length (1)
      w (1)
      l_segment (2)
      fascicles_length (4)
      fascicles_angles (4)
      branch_angle (1)
      root_time (1)
      cv (1)
    total = 2+1+1+2+4+4+1+1+1 = 17
    """
    P: list[OptimParam] = []
    P.append(OptimParam("init_length", 1.0 * np.ones(dim), 5.0 * np.ones(dim), "uniform"))
    P.append(OptimParam("length", 4.0 * np.ones(1), 12.0 * np.ones(1), "uniform"))
    P.append(OptimParam("w", 0.0 * np.ones(1), 1.0 * np.ones(1), "uniform"))
    P.append(OptimParam("l_segment", 0.1 * np.ones(dim), 3.0 * np.ones(dim), "uniform"))
    P.append(OptimParam("fascicles_length", 2.0 * np.ones(2 * dim), 10.0 * np.ones(2 * dim), "uniform"))
    P.append(OptimParam("fascicles_angles", 0.1 * np.ones(2 * dim), 1.2 * np.ones(2 * dim), "uniform"))
    P.append(OptimParam("branch_angle", 0.1 * np.ones(1), 1.3 * np.ones(1), "uniform"))
    P.append(OptimParam("root_time", -75.0 * np.ones(1), 50.0 * np.ones(1), "uniform"))
    P.append(OptimParam("cv", 2.0 * np.ones(1), 4.0 * np.ones(1), "uniform"))
    return P


def flatten_bounds(var_params: list[OptimParam]) -> np.ndarray:
    lb = np.concatenate([np.asarray(p.lower).ravel() for p in var_params]).astype(np.float32)
    ub = np.concatenate([np.asarray(p.upper).ravel() for p in var_params]).astype(np.float32)
    return np.vstack([lb, ub]).T  # (D,2)


def midpoint(bounds: np.ndarray) -> np.ndarray:
    return ((bounds[:, 0] + bounds[:, 1]) * 0.5).astype(np.float32)


def perturb(x0: np.ndarray, bounds: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    span = (bounds[:, 1] - bounds[:, 0]).astype(np.float32)
    delta = (rng.uniform(-1.0, 1.0, size=x0.size).astype(np.float32)) * (frac * span)
    x = x0 + delta
    x = np.clip(x, bounds[:, 0], bounds[:, 1])
    return x.astype(np.float32)

def json_sanitize(obj):
    """
    Convert numpy/jax scalars/arrays to plain Python types so json.dumps works.
    """
    # dict
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]

    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # jax array / array-like (ArrayImpl): try np.asarray
    try:
        arr = np.asarray(obj)
        # if it becomes an ndarray or scalar, recurse
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, np.generic):
            return arr.item()
    except Exception:
        pass

    # plain python types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # fallback: stringify (last resort)
    return str(obj)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--patient-prefix", type=Path, default=DEFAULT_PATIENT_PREFIX)
    ap.add_argument("--mesh-vtk", type=Path, default=DEFAULT_MESH_VTK)
    ap.add_argument("--fibers-vtk", type=Path, default=DEFAULT_FIBERS_VTK)
    ap.add_argument("--leadf-dir", type=Path, default=DEFAULT_LEADF_DIR)
    ap.add_argument("--map-csv", type=Path, default=DEFAULT_MAP_CSV)
    ap.add_argument("--obs-json", type=Path, default=DEFAULT_OBS_JSON)

    ap.add_argument("--lv-roots", type=int, nargs="+", default=DEFAULT_LV_ROOTS)
    ap.add_argument("--rv-roots", type=int, nargs="+", default=DEFAULT_RV_ROOTS)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--perturb-frac", type=float, default=0.15, help="fraction of (ub-lb) used to perturb x0 -> x1")

    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = args.out_dir or (args.base / "poc_17d_runs" / time.strftime("%Y%m%d_%H%M%S"))
    logger = setup_logging(out_dir)

    logger.info(f"Output: {out_dir}")

    # load+project leadfields once
    t0 = time.perf_counter()
    lf_vol = load_leadfields_volumetric(args.leadf_dir, args.map_csv)
    lf_surf, vol_id = project_leadfields_to_surface(args.mesh_vtk, lf_vol, args.patient_prefix)
    t_lf = time.perf_counter() - t0
    logger.info(f"Leadfields projected: n_surf={int(vol_id.size)} in {t_lf:.3f}s")

    # load observed ECG once
    t_obs, obs_ecg = load_obs_json(args.obs_json)
    dt_obs = float(np.median(np.diff(t_obs)))
    logger.info(f"Obs ECG loaded: n={len(t_obs)} dt~{dt_obs:.6f}s")

    # myocardium once
    from myocardial_mesh import MyocardialMesh  # import from installed/working package

    t0 = time.perf_counter()
    myo = MyocardialMesh(
        mesh_path=str(args.mesh_vtk),
        electrodes_position=None,
        fibers_path=str(args.fibers_vtk),
        device="cpu",
        conductivity_params=None,
        lead_fields_dict=lf_surf,
    )
    t_myo = time.perf_counter() - t0
    logger.info(f"MyocardialMesh init: {t_myo:.3f}s")

    # baseline tree config (these are "starting point" values; vector will override)
    cfg = BO_PurkinjeTreeConfig(
        init_length=[3.0, 3.0],   # LV, RV (dim=2) -> requerido por tu config
        N_it=6,
        length=7.0,
        w=0.0,
        l_segment=0.35,
        branch_angle=0.37,
        fascicles_angles=[0.52, 1.05],
        fascicles_length=[4.0, 4.0],
    )

    meshes_list = [args.lv_roots[0], args.lv_roots[1], args.rv_roots[0], args.rv_roots[1]]
    bo_tree = BO_PurkinjeTree(str(args.patient_prefix), meshes_list, cfg, myo)
    bo_model = BO_ecg(bo_tree)

    # 17D bounds
    var_params = var_params_17d(dim=2)
    bounds = flatten_bounds(var_params)
    x0 = midpoint(bounds)
    x1 = perturb(x0, bounds, frac=float(args.perturb_frac), rng=rng)

    # save config
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "perturb_frac": args.perturb_frac,
                "mesh_vtk": str(args.mesh_vtk),
                "fibers_vtk": str(args.fibers_vtk),
                "obs_json": str(args.obs_json),
                "lv_roots": args.lv_roots,
                "rv_roots": args.rv_roots,
                "tree_config": asdict(cfg),
                "bounds": bounds.tolist(),
                "x0_midpoint": x0.tolist(),
                "x1_perturbed": x1.tolist(),
                "var_params": [
                    {
                        "param": p.parameter,
                        "lb": np.asarray(p.lower).ravel().tolist(),
                        "ub": np.asarray(p.upper).ravel().tolist(),
                        "prior": str(p.prior),
                    }
                    for p in var_params
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = []

    def eval_once(x: np.ndarray, label: str) -> None:
        eval_dir = out_dir / label
        eval_dir.mkdir(parents=True, exist_ok=True)

        (eval_dir / "x.json").write_text(json.dumps({"x": x.tolist()}, indent=2), encoding="utf-8")

        # x -> dict params using your BO_ecg helper
        dict_params = bo_model.set_dictionary_variables(var_params, x)
        (eval_dir / "params.json").write_text(
            json.dumps(json_sanitize(dict_params), indent=2),
            encoding="utf-8",
        )

        logger.info(f"[{label}] running forward (this is the expensive part) ...")
        t0 = time.perf_counter()
        ecg_sim, _, _ = bo_tree.run_ECG(modify=True, side="both", **dict_params)
        t_forward = time.perf_counter() - t0

        # save sim npz
        sim12 = sim_to_12lead_dict(ecg_sim)
        sim_npz = eval_dir / "sim_ecg.npz"
        np.savez(sim_npz, **{k: sim12[k] for k in sim12.keys()})

        # build sim time axis (use obs dt as fallback)
        n_sim = len(sim12["I"])
        t_sim = (np.arange(n_sim, dtype=np.float32) * dt_obs).astype(np.float32)

        # metrics
        t0 = time.perf_counter()
        df, summ = qrs_metrics_lsq(t_obs, obs_ecg, t_sim, sim12)
        t_metrics = time.perf_counter() - t0

        metrics_csv = eval_dir / "metrics_qrs_lsq.csv"
        df.to_csv(metrics_csv, index=False)

        rmse_norm = float(summ["mean_rmse_norm"])
        rho = float(summ["mean_rho"])
        abs_rho = float(summ["mean_abs_rho"])

        top3 = df.sort_values("abs_rho", ascending=False).head(3)[["lead", "rho", "rmse_norm"]]
        top3.to_csv(eval_dir / "top3_by_abs_rho.csv", index=False)

        logger.info(
            f"[{label}] time_forward={t_forward:.2f}s time_metrics={t_metrics:.2f}s | "
            f"RMSE(norm)={rmse_norm:.6f} rho={rho:.6f} | top3={top3[['lead','rho']].to_dict('records')}"
        )

        summary_rows.append(
            dict(
                label=label,
                time_forward_sec=float(t_forward),
                time_metrics_sec=float(t_metrics),
                rmse_norm_mean=rmse_norm,
                rho_mean=rho,
                abs_rho_mean=abs_rho,
                sim_npz=str(sim_npz),
                metrics_csv=str(metrics_csv),
            )
        )

        # incremental save
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    # baseline x0
    eval_once(x0, "baseline_x0_midpoint")

    # one changed vector x1
    eval_once(x1, "one_step_x1_perturbed")

    logger.info("=== DONE ===")
    logger.info(f"Summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
