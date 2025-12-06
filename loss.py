from typing import Dict
import tensorflow as tf
from config import LossWeights, MaterialConfig, GeometryConfig
import physics

def pde_loss(xy_int: tf.Tensor,
        model: tf.keras.Model,
        material: MaterialConfig,
        weights: LossWeights):
    r_all = physics.compute_pde_residual(xy_int, model, material)
    r_mom = r_all[:, 0:2]  # (N, 2)
    r_cont = r_all[:, 2:3]  # (N, 1)
    loss_mom = tf.reduce_mean(tf.reduce_sum(tf.square(r_mom), axis=-1))
    loss_cont = tf.reduce_mean(tf.square(r_cont))
    return loss_mom + (weights.w_continuity * loss_cont)

def dirichlet_loss(xy_dir: tf.Tensor, model: tf.keras.Model):
    u_pred = model(xy_dir, training=True)[:, 0:2]
    u_tar = physics.dirichlet_bc_target(xy_dir)
    err = u_pred - u_tar
    return tf.reduce_mean(tf.reduce_sum(tf.square(err), axis=-1))

def neumann_loss(
        xy_neu: tf.Tensor,
        model: tf.keras.Model,
        material: MaterialConfig):
    r_trac = physics.compute_neumann_residual(
        xy_neu, model, material
    )
    return tf.reduce_mean(tf.reduce_sum(tf.square(r_trac), axis=-1))

def free_boundary_loss(
        xy_free: tf.Tensor,
        norms_free: tf.Tensor,
        model: tf.keras.Model,
        material: MaterialConfig):
    if xy_free is None or tf.shape(xy_free)[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)
    r_trac = physics.compute_traction_residual(
        xy_free, norms_free, model, material
        )
    return tf.reduce_mean(tf.reduce_sum(tf.square(r_trac), axis=-1))

def compute_loss_components(
        model: tf.keras.Model,
        coords: Dict[str, tf.Tensor],
        material: MaterialConfig,
        geom: GeometryConfig,
        weights: LossWeights):
    xy_int = coords.get("xy_int")
    xy_dir = coords.get("xy_dir")
    xy_neu = coords.get("xy_neu")
    xy_free = coords.get("xy_free")
    norms_free = coords.get("norms_free")
    lpde = pde_loss(xy_int, model, material, weights)
    ldir = dirichlet_loss(xy_dir, model)
    lneu = neumann_loss(xy_neu, model, material)
    lfree = free_boundary_loss(xy_free, norms_free, model, material)
    return {
        "pde": lpde,
        "dirichlet": ldir,
        "neumann": lneu,
        "free": lfree
    }

def total_loss(
        model: tf.keras.Model,
        coords: Dict[str, tf.Tensor],
        material: MaterialConfig,
        geom: GeometryConfig,
        loss_weights: LossWeights):
    comps = compute_loss_components(
        model, coords, material, geom, loss_weights
        )
    total = (
            loss_weights.w_pde * comps["pde"]
            + loss_weights.w_dirichlet * comps["dirichlet"]
            + loss_weights.w_neumann * comps["neumann"]
            + loss_weights.w_free * comps["free"]
    )
    return total, comps