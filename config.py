from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class MaterialConfig:
    E = 1.0
    nu = 0.499999975
    plane_strain = True

@dataclass
class GeometryConfig:
    x0 = 0.0
    x1 = 48.0
    y0_bottom = 0.0
    y1_bottom = 44.0
    y0_top = 44.0
    y1_top = 60.0

    @property
    def x_min(self) -> float:
        return self.x0
    @property
    def x_max(self) -> float:
        return self.x1
    @property
    def y_min(self) -> float:
        return self.y0_bottom
    @property
    def y_max(self) -> float:
        return self.y1_top
    def y_bottom(self, x: np.ndarray) -> np.ndarray:
        m = (self.y1_bottom - self.y0_bottom) / (self.x1 - self.x0)
        return self.y0_bottom + m * (x - self.x0)
    def y_top(self, x: np.ndarray) -> np.ndarray:
        m = (self.y1_top - self.y0_top) / (self.x1 - self.x0)
        return self.y0_top + m * (x - self.x0)

@dataclass
class NetworkConfig:
    hidden_layers = 4
    hidden_units = 128
    activation = "tanh"

@dataclass
class LossWeights:
    w_pde: float
    w_dirichlet: float
    w_neumann: float
    w_continuity: float
    w_free: float

@dataclass
class StageConfig:
    name: str
    num_epochs: int
    learning_rate: float
    n_int: int
    n_dir: int
    n_neu: int
    n_free: int
    sampling_strategy: str
    loss_weights: LossWeights = field(default_factory=LossWeights)

@dataclass
class TrainingConfig:
    stages: List[StageConfig] = field(default_factory=list)
    seed = 1234
    checkpoint_dir: str = "2/checkpoints"

@dataclass
class EvalConfig:
    grid_res_x = 121
    grid_res_y = 111
    ref_tip_disp = 16.43258437

@dataclass
class Config:
    material: MaterialConfig = field(default_factory=MaterialConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

def get_default_config():
    stage1 = StageConfig(
        name="stage_1",
        num_epochs=5000,
        learning_rate=1e-3,
        n_int=5000,
        n_dir=1000,
        n_neu=4000,
        n_free=1000,
        sampling_strategy="stress_focus",
        loss_weights=LossWeights(
            w_pde=1.0,
            w_dirichlet=100.0,
            w_neumann=1000.0,
            w_free = 0.00001,
            w_continuity=0.01
        ),
    )
    stage2 = StageConfig(
        name="stage_2_refine",
        num_epochs=5000,
        learning_rate=1e-5,
        n_int=5000,
        n_dir=1000,
        n_neu=4000,
        n_free=4000,
        sampling_strategy="stress_focus",
        loss_weights=LossWeights(
            w_pde=1.0,
            w_dirichlet=100.0,
            w_neumann=500.0,
            w_free = 100.0,
            w_continuity=0.05
        ),
    )
    cfg = Config()
    cfg.training.stages = [stage1, stage2]
    return cfg

## function for aggressive training with fourier
# def get_default_config() -> Config:
#     stage1 = StageConfig(
#         name="stage_1_aggressive",
#         num_epochs=10000,
#         learning_rate=1e-2, # <--- BOOST (0.01)
#         n_int=5000,
#         n_dir=0,
#         n_neu=1000,
#         n_free=0,
#         sampling_strategy="stress_focus",
#         loss_weights=LossWeights(
#             w_pde=1.0,
#             w_dirichlet=0.0,
#             w_neumann=10000.0, # <--- BOOST (Force the load)
#             w_free=0.0,
#             w_continuity=1.0
#         ),
#     )
#     cfg = Config()
#     cfg.training.stages = [stage1]
#     return cfg