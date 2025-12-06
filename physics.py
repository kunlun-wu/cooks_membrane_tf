import tensorflow as tf
from config import MaterialConfig, GeometryConfig

LOAD_TY = 0.0625 # Shear

def compute_pde_residual(
        xy: tf.Tensor,
        model: tf.keras.Model,
        material: MaterialConfig,):

    xy = tf.cast(xy, tf.float32)

    # Shear Modulus mu = E / (2(1+nu))
    mu = material.E / (2.0 * (1.0 + material.nu))

    # Lame parameter lambda for continuity equation: div(u) + p/lambda = 0
    nu = material.nu
    lam = (material.E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)

        # Forward Pass: Predict u, v, AND p
        outputs = model(xy, training=True)  # (N, 3)
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]  # Pressure (Mean Stress)

        # First Derivatives (Displacement) using tape.gradient (Fast)
        grad_u = tape.gradient(u, xy)  # (N, 2)
        grad_v = tape.gradient(v, xy)  # (N, 2)

        du_dx = grad_u[:, 0:1]
        du_dy = grad_u[:, 1:2]
        dv_dx = grad_v[:, 0:1]
        dv_dy = grad_v[:, 1:2]

        # Strains
        eps_xx = du_dx
        eps_yy = dv_dy
        gamma_xy = du_dy + dv_dx

        # Volumetric Strain and Deviatoric Strain
        vol_strain = eps_xx + eps_yy
        e_xx = eps_xx - (vol_strain / 3.0)
        e_yy = eps_yy - (vol_strain / 3.0)
        e_xy = 0.5 * gamma_xy

        # Stresses: Sigma = s_dev + p*I
        # Deviatoric Stress
        s_dev_xx = 2.0 * mu * e_xx
        s_dev_yy = 2.0 * mu * e_yy
        s_dev_xy = 2.0 * mu * e_xy

        # Total Stress
        sxx = s_dev_xx + p
        syy = s_dev_yy + p
        sxy = s_dev_xy

    # Second Derivatives (Momentum)
    grad_sxx = tape.gradient(sxx, xy)
    grad_syy = tape.gradient(syy, xy)
    grad_sxy = tape.gradient(sxy, xy)

    del tape

    dsxx_dx = grad_sxx[:, 0:1]
    dsxy_dy = grad_sxy[:, 1:2]
    dsxy_dx = grad_sxy[:, 0:1]
    dsyy_dy = grad_syy[:, 1:2]

    # Momentum
    mom_x = dsxx_dx + dsxy_dy
    mom_y = dsxy_dx + dsyy_dy

    # Incompressibility / Continuity Equation
    # div(u) - p/lambda = 0
    continuity = vol_strain - (p / lam)
    scale = 1.0 / LOAD_TY
    return tf.concat([mom_x * scale, mom_y * scale, continuity], axis=1)

def dirichlet_bc_target(xy: tf.Tensor):
    n = tf.shape(xy)[0]
    return tf.zeros((n, 2), dtype=tf.float32)

def compute_neumann_residual(
        xy: tf.Tensor,
        model: tf.keras.Model,
        material: MaterialConfig):
    xy = tf.cast(xy, tf.float32)
    mu = material.E / (2.0 * (1.0 + material.nu))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        outputs = model(xy, training=True)
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        grad_u = tape.gradient(u, xy)
        grad_v = tape.gradient(v, xy)
    del tape

    du_dx = grad_u[:, 0:1]
    du_dy = grad_u[:, 1:2]
    dv_dx = grad_v[:, 0:1]
    dv_dy = grad_v[:, 1:2]

    eps_xx = du_dx
    eps_yy = dv_dy
    gamma_xy = du_dy + dv_dx

    vol_strain = eps_xx + eps_yy
    e_xx = eps_xx - (vol_strain / 3.0)
    e_yy = eps_yy - (vol_strain / 3.0)
    e_xy = 0.5 * gamma_xy

    # Stress reconstruction
    sxx = (2.0 * mu * e_xx) + p
    sxy = (2.0 * mu * e_xy)

    # Right boundary traction check, normal n = (1, 0)
    tx_pred = sxx
    ty_pred = sxy

    # Target: [0, LOAD_TY]
    tx_tar = tf.zeros_like(tx_pred)
    ty_tar = tf.ones_like(ty_pred) * LOAD_TY
    scale = 1.0 / LOAD_TY
    r_x = (tx_pred - tx_tar) * scale
    r_y = (ty_pred - ty_tar) * scale
    return tf.concat([r_x, r_y], axis=1)

def compute_traction_residual(
        xy: tf.Tensor,
        normals: tf.Tensor,
        model: tf.keras.Model,
        material: MaterialConfig,):
    xy = tf.cast(xy, tf.float32)
    normals = tf.cast(normals, tf.float32)
    mu = material.E / (2.0 * (1.0 + material.nu))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        outputs = model(xy, training=True)
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        grad_u = tape.gradient(u, xy)
        grad_v = tape.gradient(v, xy)
    del tape

    # Strain
    du_dx = grad_u[:, 0:1]
    du_dy = grad_u[:, 1:2]
    dv_dx = grad_v[:, 0:1]
    dv_dy = grad_v[:, 1:2]
    eps_xx = du_dx
    eps_yy = dv_dy
    gamma_xy = du_dy + dv_dx
    vol = eps_xx + eps_yy

    # Deviatoric Strain
    e_xx = eps_xx - (vol / 3.0)
    e_yy = eps_yy - (vol / 3.0)
    e_xy = 0.5 * gamma_xy

    # Total Stress
    sxx = (2.0 * mu * e_xx) + p
    syy = (2.0 * mu * e_yy) + p
    sxy = (2.0 * mu * e_xy)

    # Traction Vector t = sigma . n
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return tf.concat([tx, ty], axis=1)

def tip_point(geom: GeometryConfig) -> tf.Tensor:
    x_tip = geom.x1
    y_tip = geom.y_top(x_tip)
    return tf.constant([[x_tip, y_tip]], dtype=tf.float32)