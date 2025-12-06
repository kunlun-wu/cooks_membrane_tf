import tensorflow as tf
from typing import Dict
from config import Config, StageConfig
from helpers import pcgrad_surgery, to_tensor
from sampling import make_sampling_sets
from loss import compute_loss_components
from physics import tip_point, compute_pde_residual

EPS = 1e-8

def _build_optimizer(initial_lr: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=initial_lr)

def train_stage(
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        cfg: Config,
        stage_cfg: StageConfig,
        fixed_scales: Dict[str, float]):

    optimizer.learning_rate.assign(stage_cfg.learning_rate)
    w = stage_cfg.loss_weights
    loss_scales = fixed_scales

    # Generate Training Data
    sampling_sets = make_sampling_sets(stage_cfg, cfg.geometry)
    xy_int = to_tensor(sampling_sets["xy_int"])
    xy_dir = to_tensor(sampling_sets["xy_dir"])
    xy_neu = to_tensor(sampling_sets["xy_neu"])
    xy_free = to_tensor(sampling_sets["xy_free"])
    norms_free = to_tensor(sampling_sets["norms_free"])

    coords = {
        "xy_int": xy_int,
        "xy_dir": xy_dir,
        "xy_neu": xy_neu,
        "xy_free": xy_free,
        "norms_free": norms_free
    }

    tip_xy = tip_point(cfg.geometry)

    history = {
        "pde": [],
        "dirichlet": [],
        "neumann": [],
        "free": [],
        "total": []
    }

    for epoch in range(1, stage_cfg.num_epochs + 1):
        with tf.GradientTape(persistent=True) as tape:
            comps = compute_loss_components(
                model=model,
                coords=coords,
                material=cfg.material,
                geom=cfg.geometry,
                weights=w,
            )

        vars_ = model.trainable_variables
        grads_tasks = []

        # 1. Effective Weights (Normalized)
        w_pde_eff = w.w_pde / (loss_scales["pde"] + EPS)
        w_dir_eff = w.w_dirichlet / (loss_scales["dirichlet"] + EPS)
        w_neu_eff = w.w_neumann / (loss_scales["neumann"] + EPS)
        w_free_eff = w.w_free / (loss_scales.get("free", 1.0) + EPS)

        # 2. Gradient Calculation
        if w_pde_eff > 0.0:
            g = tape.gradient(comps["pde"], vars_)
            grads_tasks.append(
                [w_pde_eff * gi if gi is not None else None for gi in g]
                )
        if w_dir_eff > 0.0:
            g = tape.gradient(comps["dirichlet"], vars_)
            grads_tasks.append(
                [w_dir_eff * gi if gi is not None else None for gi in g]
                )
        if w_neu_eff > 0.0:
            g = tape.gradient(comps["neumann"], vars_)
            grads_tasks.append(
                [w_neu_eff * gi if gi is not None else None for gi in g]
                )
        if w_free_eff > 0.0:
            g = tape.gradient(comps["free"], vars_)
            grads_tasks.append(
                [w_free_eff * gi if gi is not None else None for gi in g]
                )
        del tape

        # 3. Gradient Surgery & Update
        combined_grads = pcgrad_surgery(grads_tasks)
        optimizer.apply_gradients(zip(combined_grads, vars_))

        # 4. Logging & Monitoring
        loss_total = (
                w.w_pde * comps["pde"] +
                w.w_dirichlet * comps["dirichlet"] +
                w.w_neumann * comps["neumann"] +
                w.w_free * comps["free"]
        )
        history["pde"].append(float(comps["pde"]))
        history["dirichlet"].append(float(comps["dirichlet"]))
        history["neumann"].append(float(comps["neumann"]))
        history["free"].append(float(comps["free"]))
        history["total"].append(float(loss_total))

        if epoch % 100 == 0 or epoch == 1 or epoch == stage_cfg.num_epochs:
            # A. Normalized Losses (Diagnostic)
            pde_norm = comps["pde"] / loss_scales["pde"]
            dir_norm = comps["dirichlet"] / loss_scales["dirichlet"]
            neu_norm = comps["neumann"] / loss_scales["neumann"]
            free_norm = comps["free"] / loss_scales.get("free", 1.0)

            # B. Tip Displacement (Physical)
            # Evaluate model at tip (training=False to be safe)
            tip_pred = model(tip_xy, training=False)
            u_tip_y = float(tip_pred[0, 1])

            # C. Continuity vs Momentum Breakdown
            r_all = compute_pde_residual(xy_int, model, cfg.material)
            loss_mom = tf.reduce_mean(
                tf.reduce_sum(tf.square(r_all[:, 0:2]), axis=1)
                )
            loss_cont = tf.reduce_mean(tf.square(r_all[:, 2:3]))

            print(
                f"[{stage_cfg.name}]Ep {epoch} | "
                f"Tip: {u_tip_y:.4f} | "
                f"PDE: {pde_norm:.1e} (Mom: {loss_mom:.1e}, Cont: {loss_cont:.1e}) | "
                f"Neu: {neu_norm:.1e} | "
                f"Dir: {dir_norm:.1e} | "
                f"Free: {free_norm:.1e}"
            )
    return history


def train_all(model: tf.keras.Model, cfg: Config):
    first_lr = cfg.training.stages[0].learning_rate
    optimizer = _build_optimizer(first_lr)
    print("--- Computing Initial Loss Scales (Normalization) ---")
    init_stage = cfg.training.stages[0]
    sets = make_sampling_sets(init_stage, cfg.geometry)
    coords = {
        "xy_int": to_tensor(sets["xy_int"]),
        "xy_dir": to_tensor(sets["xy_dir"]),
        "xy_neu": to_tensor(sets["xy_neu"]),
        "xy_free": to_tensor(sets["xy_free"]),
        "norms_free": to_tensor(sets["norms_free"])
    }
    from config import LossWeights
    dummy_weights = LossWeights(1.0, 1.0, 1.0, 1.0, 1.0)
    comps0 = compute_loss_components(
        model=model, coords=coords, material=cfg.material,
        geom=cfg.geometry, weights=dummy_weights
    )
    # Define a minimum scale to prevent division by zero if loss is 0.0
    MIN_SCALE = 1e-8
    fixed_scales = {
        "pde": float(comps0["pde"]) if float(
            comps0["pde"]
            ) > MIN_SCALE else 1.0,
        "dirichlet": float(comps0["dirichlet"]) if float(
            comps0["dirichlet"]
            ) > MIN_SCALE else 1.0,
        "neumann": float(comps0["neumann"]) if float(
            comps0["neumann"]
            ) > MIN_SCALE else 1.0,
        "free": float(comps0["free"]) if float(
            comps0["free"]
            ) > MIN_SCALE else 1.0,
    }
    print(f"Fixed Normalization Scales: {fixed_scales}")

    global_history = {
        "pde": [],
        "dirichlet": [],
        "neumann": [],
        "free": [],
        "total": []
    }

    for stage_cfg in cfg.training.stages:
        print(f"--- Training stage: {stage_cfg.name} ---")
        stage_hist = train_stage(
            model, optimizer, cfg, stage_cfg, fixed_scales
            )
        for key in global_history:
            if key in stage_hist:
                global_history[key].extend(stage_hist[key])

    return model, optimizer, global_history