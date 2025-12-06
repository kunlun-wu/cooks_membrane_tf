import tensorflow as tf
from config import NetworkConfig

class PINN(tf.keras.Model):
    """MLP mapping (x, y) to (u_x, u_y, p)."""
    def __init__(self, net_cfg: NetworkConfig):
        super().__init__()
        self.x_mean = tf.constant(24.0, dtype=tf.float32)
        self.y_mean = tf.constant(30.0, dtype=tf.float32)
        self.x_scale = tf.constant(24.0, dtype=tf.float32)
        self.y_scale = tf.constant(30.0, dtype=tf.float32)
        self.layers_hidden = []
        for _ in range(net_cfg.hidden_layers):
            self.layers_hidden.append(
                tf.keras.layers.Dense(
                    net_cfg.hidden_units, activation=net_cfg.activation
                    )
            )
        self.out = tf.keras.layers.Dense(3, activation=None)

    def call(self, xy, training=False):
        xy = tf.cast(xy, tf.float32)
        # Normalize inputs
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        x_n = (x - self.x_mean) / self.x_scale
        y_n = (y - self.y_mean) / self.y_scale
        z = tf.concat([x_n, y_n], axis=1)
        # Pass through hidden layers
        for layer in self.layers_hidden:
            z = layer(z)

        # return self.out(z)

        outputs = self.out(z)
        u = outputs[:, 0:1] * 20.0
        v = outputs[:, 1:2] * 20.0
        p = outputs[:, 2:3] * 1.0

        return tf.concat([u, v, p], axis=1)

def build_model(net_cfg: NetworkConfig):
    return PINN(net_cfg)

# # WITH BETTER SCALING AND HARD CONSTRAINT
# import tensorflow as tf
# from config import NetworkConfig
#
# class PINN(tf.keras.Model):
#     def __init__(self, net_cfg: NetworkConfig):
#         super().__init__()
#         self.layers_hidden = []
#         for _ in range(net_cfg.hidden_layers):
#             self.layers_hidden.append(
#                 tf.keras.layers.Dense(
#                     net_cfg.hidden_units, activation=net_cfg.activation
#                 )
#             )
#         self.out = tf.keras.layers.Dense(
#             3,
#             activation=None,
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
#             bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
#         )
#     def call(self, xy, training=False):
#         xy = tf.cast(xy, tf.float32)
#         x = xy[:, 0:1]
#         y = xy[:, 1:2]
#
#         # 1. ISOPARAMETRIC TRANSFORMATION
#         # Maps Cook's Trapezoid to Reference Square [-1, 1]
#
#         # Map x: [0, 48] -> [-1, 1]
#         x_t = (x - 24.0) / 24.0
#
#         # Map y: Flattens the slanted top/bottom edges
#         # Formula: y_t = 3 * (5x - 8y + 176) / (7x - 528)
#         num = 3.0 * (5.0 * x - 8.0 * y + 176.0)
#         den = 7.0 * x - 528.0
#         y_t = num / den
#
#         z = tf.concat([x_t, y_t], axis=1)
#
#         # 2. MLP PASS
#         for layer in self.layers_hidden:
#             z = layer(z)
#         outputs = self.out(z)
#         u_raw = outputs[:, 0:1]
#         v_raw = outputs[:, 1:2]
#         p = outputs[:, 2:3]
#
#         # 3. HARD DIRICHLET CONSTRAINT + SCALING
#         # Enforce u=0, v=0 at x=0 explicitly.
#         # factor is 0.0 at x=0, and 1.0 at x=48.
#         factor = x / 48.0
#
#         u = u_raw * factor * 20.0
#         v = v_raw * factor * 20.0
#         p = p * 1.0
#         return tf.concat([u, v, p], axis=1)
#
# def build_model(net_cfg: NetworkConfig):
#     return PINN(net_cfg)

# WITH FOURIER FEATURES, BETTER SCALING, AND HARD CONSTRAINT
# import tensorflow as tf
# import numpy as np
# from config import NetworkConfig
#
# class FourierEmbedding(tf.keras.layers.Layer):
#     def __init__(self, num_features=128, scale=2.0):
#         super().__init__()
#         self.num_features = num_features
#         self.scale = scale
#         self.B = None
#
#     def build(self, input_shape):
#         dim = input_shape[-1]
#         self.B = tf.random.normal(
#             shape=(dim, self.num_features // 2),
#             mean=0.0, stddev=self.scale, dtype=tf.float32
#         )
#         self.B = tf.Variable(self.B, trainable=False)
#
#     def call(self, x):
#         x_proj = 2.0 * np.pi * tf.matmul(x, self.B)
#         return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)
#
# class PINN(tf.keras.Model):
#     def __init__(self, net_cfg: NetworkConfig):
#         super().__init__()
#
#         # Fourier Embedding
#         self.embedding = FourierEmbedding(
#             num_features=net_cfg.hidden_units, scale=1.0
#             )
#
#         # Hidden Layers
#         self.layers_hidden = []
#         for _ in range(net_cfg.hidden_layers):
#             self.layers_hidden.append(
#                 tf.keras.layers.Dense(
#                     net_cfg.hidden_units, activation=net_cfg.activation
#                 )
#             )
#
#         # Output Layer (Small Random Init to break symmetry)
#         self.out = tf.keras.layers.Dense(
#             3,
#             activation=None,
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
#             bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
#         )
#
#     def call(self, xy, training=False):
#         xy = tf.cast(xy, tf.float32)
#         x = xy[:, 0:1]
#         y = xy[:, 1:2]
#
#         # 1. ISOPARAMETRIC TRANSFORMATION
#         x_t = (x - 24.0) / 24.0
#         num = 3.0 * (5.0 * x - 8.0 * y + 176.0)
#         den = 7.0 * x - 528.0
#         y_t = num / den
#         z = tf.concat([x_t, y_t], axis=1)
#
#         # 2. FOURIER FEATURES
#         # Apply Fourier features to the "Square" domain inputs
#         z = self.embedding(z)
#
#         # 3. MLP
#         for layer in self.layers_hidden:
#             z = layer(z)
#         outputs = self.out(z)
#         u_raw = outputs[:, 0:1]
#         v_raw = outputs[:, 1:2]
#         p = outputs[:, 2:3]
#
#         # 4. HARD DIRICHLET CONSTRAINT
#         factor = x / 48.0
#         u = u_raw * factor * 20.0
#         v = v_raw * factor * 20.0
#         p = p * 1.0
#         return tf.concat([u, v, p], axis=1)
#
# def build_model(net_cfg: NetworkConfig):
#     return PINN(net_cfg)