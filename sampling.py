import numpy as np
from config import GeometryConfig, StageConfig

def sample_interior_uniform(n_points: int, geom: GeometryConfig):
    x = np.random.uniform(geom.x0, geom.x1, size=(n_points, 1))
    y_bot = geom.y_bottom(x)
    y_top = geom.y_top(x)
    y = np.random.uniform(y_bot, y_top)
    return np.hstack([x, y])

def sample_interior_tip_focus(
        n_points: int,
        geom: GeometryConfig,
        focus_frac: float = 0.7,
        x_focus_ratio: float = 0.3):
    n_focus = int(n_points * focus_frac)
    n_uniform = n_points - n_focus
    if n_uniform < 0:
        pts_uniform = sample_interior_uniform(n_uniform, geom)
    else:
        pts_uniform = np.empty((0, 2))
    x_min_f = geom.x1 - x_focus_ratio * (geom.x1 - geom.x0)
    x_max_f = geom.x1
    x_f = np.random.uniform(x_min_f, x_max_f, size=(n_focus, 1))
    y_bot_f = geom.y_bottom(x_f)
    y_top_f = geom.y_top(x_f)
    y_f = np.random.uniform(y_bot_f, y_top_f)
    pts_focus = np.hstack([x_f, y_f])
    if pts_uniform.size == 0:
        return pts_focus
    return np.vstack([pts_uniform, pts_focus])

def sample_interior_corner_focus(
        n_points: int,
        geom: GeometryConfig,
        radius: float = 5.0):
    xc, yc = geom.x0, geom.y0_top  # (0.0, 44.0)
    points = []
    while len(points) < n_points:
        r = np.random.uniform(0, radius, size=n_points * 2)
        theta = np.random.uniform(-np.pi / 2, 0, size=n_points * 2)
        x_prop = xc + r * np.cos(theta)  # x > 0
        y_prop = yc + r * np.sin(theta)  # y < 44
        y_bot_limit = geom.y_bottom(x_prop)
        mask = (x_prop >= geom.x0) & (x_prop <= geom.x1) & \
               (y_prop >= y_bot_limit) & (y_prop <= geom.y_top(x_prop))
        valid_x = x_prop[mask]
        valid_y = y_prop[mask]
        for vx, vy in zip(valid_x, valid_y):
            if len(points) < n_points:
                points.append([vx, vy])
            else:
                break
    return np.array(points)

def sample_boundary_dirichlet(n_points: int, geom: GeometryConfig):
    x = np.full((n_points, 1), geom.x0)
    y_bot = geom.y_bottom(x)  # 0 at x=0
    y_top = geom.y_top(x)     # 44 at x=0
    y = np.random.uniform(y_bot, y_top)
    return np.hstack([x, y])

def sample_boundary_neumann(n_points: int, geom: GeometryConfig):
    x = np.full((n_points, 1), geom.x1)
    y_bot = geom.y_bottom(x)  # 44 at x=48
    y_top = geom.y_top(x)     # 60 at x=48
    y = np.random.uniform(y_bot, y_top)
    return np.hstack([x, y])

def sample_interior(
        n_points: int,
        geom: GeometryConfig,
        strategy: str = "uniform"):
    if n_points <= 0:
        return np.empty((0, 2))
    if strategy == "stress_focus":
        n_tip = int(n_points * 0.5)
        n_corner = int(n_points * 0.3)
        n_unif = n_points - n_tip - n_corner
        pts_tip = sample_interior_tip_focus(n_tip, geom)
        pts_corner = sample_interior_corner_focus(n_corner, geom)
        pts_unif = sample_interior_uniform(n_unif, geom)
        return np.vstack([pts_tip, pts_corner, pts_unif])
    return sample_interior_uniform(n_points, geom)

def sample_boundary_free(n_points: int, geom: GeometryConfig):
    """Sample points on Top and Bottom edges."""
    if n_points <= 0:
        return np.empty((0, 2)), np.empty((0, 2))
    n_top = n_points // 2
    n_bot = n_points - n_top
    x_top = np.random.uniform(geom.x0, geom.x1, (n_top, 1))
    y_top = geom.y_top(x_top)
    pts_top = np.hstack([x_top, y_top])
    nx_top = -1.0 / np.sqrt(10.0)
    ny_top = 3.0 / np.sqrt(10.0)
    norms_top = np.tile([nx_top, ny_top], (n_top, 1))
    x_bot = np.random.uniform(geom.x0, geom.x1, (n_bot, 1))
    y_bot = geom.y_bottom(x_bot)
    pts_bot = np.hstack([x_bot, y_bot])
    # Calculate magnitude for normalization
    mag_bot = np.sqrt(44.0 ** 2 + 48.0 ** 2)
    nx_bot = 44.0 / mag_bot
    ny_bot = -48.0 / mag_bot
    norms_bot = np.tile([nx_bot, ny_bot], (n_bot, 1))
    # Combine
    xy = np.vstack([pts_top, pts_bot])
    normals = np.vstack([norms_top, norms_bot])
    return xy, normals

def make_sampling_sets(stage_cfg: StageConfig, geom: GeometryConfig):
    xy_int = sample_interior(
        stage_cfg.n_int, geom, stage_cfg.sampling_strategy
        )
    xy_dir = sample_boundary_dirichlet(stage_cfg.n_dir, geom)
    xy_neu = sample_boundary_neumann(stage_cfg.n_neu, geom)

    # Sample free boundaries with normals
    xy_free, norms_free = sample_boundary_free(stage_cfg.n_free, geom)
    return {
        "xy_int": xy_int,
        "xy_dir": xy_dir,
        "xy_neu": xy_neu,
        "xy_free": xy_free,  # Points
        "norms_free": norms_free  # Normals
    }

# Boundary Layer Sampling for Distribution of Stress into Interior
# def sample_boundary_layer(
#         n_points: int,
#         geom: GeometryConfig,
#         width: float = 1.0):
#
#     # This bridges the gap between interior and Neumann edge
#     x_min = geom.x1 - width
#     x = np.random.uniform(x_min, geom.x1, size=(n_points, 1))
#     y_bot = geom.y_bottom(x)
#     y_top = geom.y_top(x)
#     y = np.random.uniform(y_bot, y_top)
#     return np.hstack([x, y])
#
# def make_sampling_sets(
#         stage_cfg: StageConfig,
#         geom: GeometryConfig,
#         model: Optional[object] = None,
#         material: Optional[object] = None,):
#
#     xy_int_main = sample_interior(
#         stage_cfg.n_int, geom, stage_cfg.sampling_strategy
#     )
#
#     # Add Boundary Layer
#     xy_layer = sample_boundary_layer(2000, geom, width=1.0)
#
#     # Combine
#     xy_int = np.vstack([xy_int_main, xy_layer])
#
#     xy_dir = sample_boundary_dirichlet(stage_cfg.n_dir, geom)
#     xy_neu = sample_boundary_neumann(stage_cfg.n_neu, geom)
#     xy_free, norms_free = sample_boundary_free(stage_cfg.n_free, geom)
#
#     return {
#         "xy_int": xy_int,
#         "xy_dir": xy_dir,
#         "xy_neu": xy_neu,
#         "xy_free": xy_free,
#         "norms_free": norms_free
#     }