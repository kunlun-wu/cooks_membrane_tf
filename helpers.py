import os
import random
from typing import List, Optional
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def to_tensor(x, dtype=tf.float32):
    return tf.convert_to_tensor(x, dtype=dtype)

def pcgrad_surgery(grad_lists: List[List[Optional[tf.Tensor]]]):
    if not grad_lists:
        return []

    n_tasks = len(grad_lists)
    n_vars = len(grad_lists[0])
    eps = 1e-12

    # Copy grads
    proj_grads = [
        [None if g is None else tf.identity(g) for g in grads]
        for grads in grad_lists
    ]

    # Pairwise projections
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                continue

            # Dot product g_i · g_j
            dot_terms = []
            for gi, gj in zip(proj_grads[i], proj_grads[j]):
                if gi is not None and gj is not None:
                    dot_terms.append(tf.reduce_sum(gi * gj))
            if not dot_terms:
                continue

            dot_ij = tf.add_n(dot_terms)

            # If conflicting, project g_i
            if dot_ij < 0.0:
                norm_terms = []
                for gj in proj_grads[j]:
                    if gj is not None:
                        norm_terms.append(tf.reduce_sum(gj * gj))
                if not norm_terms:
                    continue

                norm_sq_j = tf.add_n(norm_terms) + eps
                coeff = dot_ij / norm_sq_j

                new_grads_i = []
                for gi, gj in zip(proj_grads[i], proj_grads[j]):
                    if gi is None or gj is None:
                        new_grads_i.append(gi)
                    else:
                        new_grads_i.append(gi - coeff * gj)
                proj_grads[i] = new_grads_i

    # Aggregate across tasks (simple mean)
    combined: List[Optional[tf.Tensor]] = []
    for var_idx in range(n_vars):
        gv = [proj_grads[t][var_idx] for t in range(n_tasks) if proj_grads[t][var_idx] is not None]
        if not gv:
            combined.append(None)
        else:
            combined.append(tf.add_n(gv) / float(len(gv)))
    return combined

def save_checkpoint(
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    manager.save()
    return manager

def restore_latest_checkpoint(
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        ckpt_dir: str):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
    return manager.latest_checkpoint