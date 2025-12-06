import os
from typing import Dict, Optional
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import Config
from physics import tip_point

# Geometry
def _line_y(x: np.ndarray, x0: float, y0: float, x1: float, y1: float):
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)

def _bottom_top_y(X: np.ndarray, geom):
    y_bot = _line_y(X, geom.x0, geom.y0_bottom, geom.x1, geom.y1_bottom)
    y_top = _line_y(X, geom.x0, geom.y0_top, geom.x1, geom.y1_top)
    return y_bot, y_top

# Core evaluation functions
def evaluate_on_grid(model: tf.keras.Model, cfg: Config):
    """Evaluate displacement and stress on a Cook's membrane grid."""
    geom = cfg.geometry
    eval_cfg = cfg.eval
    x = np.linspace(geom.x0, geom.x1, eval_cfg.grid_res_x)
    t = np.linspace(0.0, 1.0, eval_cfg.grid_res_y)
    X, T = np.meshgrid(x, t)
    y_bot, y_top = _bottom_top_y(X, geom)
    Y = y_bot + T * (y_top - y_bot)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    xy_tf = tf.convert_to_tensor(xy, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy_tf)
        outputs = model(xy_tf, training=False)
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        # Calculate gradients for stress recovery
        grad_u = tape.gradient(u, xy_tf)
        grad_v = tape.gradient(v, xy_tf)
    del tape

    du_dx = grad_u[:, 0:1]
    du_dy = grad_u[:, 1:2]
    dv_dx = grad_v[:, 0:1]
    dv_dy = grad_v[:, 1:2]

    # Strain calc
    eps_xx = du_dx
    eps_yy = dv_dy
    gamma_xy = du_dy + dv_dx
    vol_strain = eps_xx + eps_yy

    # Deviatoric calc
    e_xx = eps_xx - vol_strain / 3.0
    e_yy = eps_yy - vol_strain / 3.0
    e_xy = 0.5 * gamma_xy

    # Material
    E = cfg.material.E
    nu = cfg.material.nu
    mu = E / (2.0 * (1.0 + nu))

    # Stress = s_dev + p*I
    sxx = (2.0 * mu * e_xx) + p
    syy = (2.0 * mu * e_yy) + p
    sxy = (2.0 * mu * e_xy)

    # Von Mises
    szz = (2.0 * mu * (-vol_strain / 3.0)) + p
    sigma_vm = tf.sqrt(
        0.5 * (
                tf.square(sxx - syy) +
                tf.square(syy - szz) +
                tf.square(szz - sxx) +
                6.0 * tf.square(sxy)
        )
    )
    shape = X.shape
    return {
        "X": X,
        "Y": Y,
        "ux": u.numpy().reshape(shape),
        "uy": v.numpy().reshape(shape),
        "p": p.numpy().reshape(shape),
        "sxx": sxx.numpy().reshape(shape),
        "syy": syy.numpy().reshape(shape),
        "sxy": sxy.numpy().reshape(shape),
        "von_mises": sigma_vm.numpy().reshape(shape),
    }

def compute_tip_displacement(model: tf.keras.Model, cfg: Config):
    tip_xy = tip_point(cfg.geometry)
    outputs = model(tip_xy, training=False)
    return outputs.numpy()[0, :2]

def compute_tip_error(u_tip_y: float, cfg: Config):
    ref = cfg.eval.ref_tip_disp
    return float(abs(u_tip_y - ref) / abs(ref) * 100.0)

# Plotting
def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_deformed_shape(
        X: np.ndarray,
        Y: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        geom,
        scale: float = 1.0,
        save_path: Optional[str] = None):

    # original trapezoid corners
    x0, x1 = geom.x0, geom.x1
    yb0, yb1 = geom.y0_bottom, geom.y1_bottom
    yt0, yt1 = geom.y0_top, geom.y1_top
    x_poly = [x0, x1, x1, x0, x0]
    y_poly = [yb0, yb1, yt1, yt0, yb0]

    # deformed boundary from grid
    Xb, Yb = X[0, :], Y[0, :]
    Xt, Yt = X[-1, :], Y[-1, :]
    Xl, Yl = X[:, 0], Y[:, 0]
    Xr, Yr = X[:, -1], Y[:, -1]

    Ubx, Uby = ux[0, :], uy[0, :]
    Utx, Uty = ux[-1, :], uy[-1, :]
    Ulx, Uly = ux[:, 0], uy[:, 0]
    Urx, Ury = ux[:, -1], uy[:, -1]

    Xb_def = Xb + scale * Ubx
    Yb_def = Yb + scale * Uby
    Xt_def = Xt + scale * Utx
    Yt_def = Yt + scale * Uty
    Xl_def = Xl + scale * Ulx
    Yl_def = Yl + scale * Uly
    Xr_def = Xr + scale * Urx
    Yr_def = Yr + scale * Ury

    plt.figure()
    plt.plot(x_poly, y_poly, linestyle="--", label="Original")
    plt.plot(Xb_def, Yb_def, color="tab:orange", label="Deformed")
    plt.plot(Xt_def, Yt_def, color="tab:orange")
    plt.plot(Xl_def, Yl_def, color="tab:orange")
    plt.plot(Xr_def, Yr_def, color="tab:orange")
    plt.axis("equal")

    dx = x1 - x0
    y_min = min(yb0, yt0)

    plt.xlim(x0 - 0.1 * dx, x1 + 0.1 * dx)
    plt.ylim(y_min - 10.0, 80.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Original vs deformed shape (scale={scale})")
    plt.legend()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def _plot_field(
        X: np.ndarray,
        Y: np.ndarray,
        F: np.ndarray,
        title: str,
        save_path: Optional[str] = None):
    plt.figure()
    cf = plt.contourf(X, Y, F, levels=50)
    plt.colorbar(cf)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_displacement_magnitude(
        X: np.ndarray,
        Y: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        save_path: Optional[str] = None):
    umag = np.sqrt(ux ** 2 + uy ** 2)
    _plot_field(X, Y, umag, "|u|", save_path)

def plot_von_mises(
        X: np.ndarray,
        Y: np.ndarray,
        vm: np.ndarray,
        save_path: Optional[str] = None):
    _plot_field(X, Y, vm, "von Mises stress", save_path)

def plot_stress_components(
        X: np.ndarray,
        Y: np.ndarray,
        sxx: np.ndarray,
        syy: np.ndarray,
        sxy: np.ndarray,
        out_dir: Optional[str] = None):
    if out_dir is None:
        out_dir = "figs"
    _plot_field(X, Y, sxx, "σ_xx", os.path.join(out_dir, "stress_sxx.png"))
    _plot_field(X, Y, syy, "σ_yy", os.path.join(out_dir, "stress_syy.png"))
    _plot_field(X, Y, sxy, "σ_xy", os.path.join(out_dir, "stress_sxy.png"))

def plot_collocation_cloud(
        coords: Dict[str, np.ndarray],
        geom,
        save_path: Optional[str] = None):
    xy_int = coords.get("xy_int")
    xy_dir = coords.get("xy_dir")
    xy_neu = coords.get("xy_neu")
    x0, x1 = geom.x0, geom.x1
    yb0, yb1 = geom.y0_bottom, geom.y1_bottom
    yt0, yt1 = geom.y0_top, geom.y1_top
    x_poly = [x0, x1, x1, x0, x0]
    y_poly = [yb0, yb1, yt1, yt0, yb0]
    plt.figure()
    plt.plot(x_poly, y_poly, "k--", linewidth=1.0, label="Boundary")
    if xy_int is not None and xy_int.size > 0:
        plt.scatter(
            xy_int[:, 0], xy_int[:, 1], s=5, alpha=0.4, label="Interior"
            )
    if xy_dir is not None and xy_dir.size > 0:
        plt.scatter(
            xy_dir[:, 0], xy_dir[:, 1], s=10, marker="s", label="Dirichlet"
            )
    if xy_neu is not None and xy_neu.size > 0:
        plt.scatter(
            xy_neu[:, 0], xy_neu[:, 1], s=10, marker="^", label="Neumann"
            )
    plt.axis("equal")
    dx = x1 - x0
    y_min = min(yb0, yt0)
    y_max = max(yb1, yt1)
    dy = y_max - y_min
    plt.xlim(x0 - 0.1 * dx, x1 + 0.1 * dx)
    plt.ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Collocation points (interior + boundaries)")
    plt.legend(loc="best")
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_evaluation_grid_mesh(
        X: np.ndarray,
        Y: np.ndarray,
        geom,
        save_path: Optional[str] = None,
):
    x0, x1 = geom.x0, geom.x1
    yb0, yb1 = geom.y0_bottom, geom.y1_bottom
    yt0, yt1 = geom.y0_top, geom.y1_top
    x_poly = [x0, x1, x1, x0, x0]
    y_poly = [yb0, yb1, yt1, yt0, yb0]
    plt.figure()

    # grid lines (rows)
    for i in range(X.shape[0]):
        plt.plot(X[i, :], Y[i, :], linewidth=0.3)
    # grid lines (cols)
    for j in range(X.shape[1]):
        plt.plot(X[:, j], Y[:, j], linewidth=0.3)
    plt.plot(x_poly, y_poly, "k--", linewidth=1.0)
    plt.axis("equal")
    dx = x1 - x0
    y_min = min(yb0, yt0)
    y_max = max(yb1, yt1)
    dy = y_max - y_min
    plt.xlim(x0 - 0.1 * dx, x1 + 0.1 * dx)
    plt.ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Evaluation grid (Cook's membrane)")
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_training_history(
        history: Dict[str, np.ndarray],
        save_path: Optional[str] = None):
    epochs = history.get("epoch", np.arange(len(history.get("total", []))))
    plt.figure()
    if "pde" in history:
        plt.plot(epochs, history["pde"], label="PDE")
    if "dirichlet" in history:
        plt.plot(epochs, history["dirichlet"], label="Dirichlet")
    if "neumann" in history:
        plt.plot(epochs, history["neumann"], label="Neumann")
    if "free" in history:
        plt.plot(epochs, history["free"], label="Free BC")
    if "total" in history:
        plt.plot(epochs, history["total"], label="Total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# Full evaluation pipeline
def run_full_evaluation(
        model: tf.keras.Model,
        cfg: Config,
        out_dir: str = "figs",
        deformed_scale: float = 1.0,
        history: Optional[Dict[str, np.ndarray]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,
):
    _ensure_dir(out_dir)
    fields = evaluate_on_grid(model, cfg)
    X = fields["X"]
    Y = fields["Y"]
    ux = fields["ux"]
    uy = fields["uy"]
    vm = fields["von_mises"]
    sxx = fields["sxx"]
    syy = fields["syy"]
    sxy = fields["sxy"]
    X_def = X + ux
    Y_def = Y + uy

    # Collocation point cloud
    plot_collocation_cloud(
        coords,
        cfg.geometry,
        save_path=os.path.join(out_dir, "collocation_cloud.png"),
    )

    # Evaluation grid mesh
    plot_evaluation_grid_mesh(
        X,
        Y,
        cfg.geometry,
        save_path=os.path.join(out_dir, "evaluation_grid.png"),
    )

    # Deformed shape and field plots
    plot_deformed_shape(
        X, Y, ux, uy, cfg.geometry,
        scale=deformed_scale,
        save_path=os.path.join(out_dir, "deformed_shape.png"),
    )

    plot_displacement_magnitude(
        X_def, Y_def, ux, uy,  # Changed X, Y to X_def, Y_def
        save_path=os.path.join(out_dir, "disp_magnitude.png"),
    )
    plot_von_mises(
        X_def, Y_def, vm,  # Changed X, Y to X_def, Y_def
        save_path=os.path.join(out_dir, "von_mises.png"),
    )
    plot_stress_components(
        X_def, Y_def, sxx, syy, sxy,  # Changed X, Y to X_def, Y_def
        out_dir=out_dir,
    )

    if history is not None:
        plot_training_history(
            history,
            save_path=os.path.join(out_dir, "training_history.png"),
        )

    u_tip = compute_tip_displacement(model, cfg)
    err = compute_tip_error(u_tip_y=u_tip[1], cfg=cfg)
    return {
        "u_tip_x": float(u_tip[0]),
        "u_tip_y": float(u_tip[1]),
        "tip_error_percent": err,
    }