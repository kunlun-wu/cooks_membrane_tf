# Cook's Membrane Problem with PINN and Adaptive Sampling using RL
The ReadMe file is written as the deliverable for course project in "Computational Solid Mechanics with AI." It is separated into two main sections: (1) Implementation of PINN for Cook's Membrane Problem (midterm project), and (2) Adaptive Sampling using Reinforcement Learning (final project). TensorFlow is used as the main framework for implementing the PINN and RL algorithms.

Cook's membrane problem is a 2D plane-stress/strain benchmark problem that involves simulating the deformation of a tapered cantilever under shear loading at the free end. The goal of this project is to solve the displacement field using PINN by enforcing equilibrium equations, boundary conditions, and constitutive laws within the loss function, and to enhance the training process through adaptive sampling strategies driven by RL. For this project, I refer to a linear elastic model with E = 1.0 and ν = 0.499999975.

## Section 1: Implementation of PINN for Cook's Membrane Problem
### Part A: Understanding the Physics
#### 1. Derivation of 2D Elasticity Equations (Navier Equations)
**Assumptions:**
* **2D Plane Strain:** We assume the deformation is uniform along the thickness z-axis.
* **Linear Elasticity:** The material behaves linearly under small deformations.
* **Isotropic Material:** Material properties are the same in all directions.
* **Near-Incompressibility:** The Poisson's ratio (nu) is set to approximately 0.5 (0.499999975).
* **No Body Forces:** Gravity and other body forces are neglected.

Navier Equations:

The strong form of the problem is governed by the balance of linear momentum (Equilibrium Equations). Since the problem is static, the sum of forces must equal zero.

**Vector Form:**

div(sigma) + b = 0

**Component Form (2D Cartesian):**

d(sigma_xx)/dx + d(sigma_xy)/dy = 0

d(sigma_yx)/dx + d(sigma_yy)/dy = 0

* **sigma:** Stress tensor.
* **b:** Body force vector (assumed to be 0 here).
* 
#### 2. Defining Linear Elastic Constitutive Relation (Hooke's Law)
**Mixed Formulation (Implemented Strategy):**
Instead of solving only for displacements (u, v), the network outputs (u, v, p), where 'p' is the hydrostatic pressure.

**Stress-Strain Relationship:**

sigma = s_dev + p * I

* **s_dev:** Deviatoric stress (shear distortion).
* **p:** Hydrostatic pressure (volume change resistance).
* **I:** Identity matrix.

**Calculating Deviatoric Stress:**

s_dev = 2 * mu * epsilon_dev

* **mu:** Shear modulus, calculated as E / (2 * (1 + nu)).
* **epsilon_dev:** Deviatoric strain, derived from the displacement gradients.

**Continuity Equation (Incompressibility Constraint):**

To enforce the material behavior, we add a constraint relating volumetric strain to pressure:

div(u) - p / lambda = 0

* **u:** Strain.
* **lambda:** Lame's first parameter. As nu approaches 0.5, lambda approaches infinity, forcing div(u) to approach 0 (incompressible).

### Part B: PINN Formulation
#### 1. Total Loss Function
**The physical equations for calculating losses are implemented in ```physics.py```, and the total loss function is defined in ```loss.py```**

The Physics-Informed Neural Network is trained by minimizing a composite loss function that simultaneously enforces the physical laws inside the domain and the conditions on the boundaries. The total loss is a weighted sum of four components:

**L_total = (w_pde * L_pde) + (w_dir * L_dirichlet) + (w_neu * L_neumann) + (w_free * L_free)**

(the subscript "free" refers to upper and lower slanted boundaries, as the resulted deformation had no curvature on these edges, this loss was implemented hoping to make the resulting deformation more accurate.)

a. Equilibrium & Constitutive Residuals (L_pde)

We minimize the residuals of the momentum balance and the continuity equation, which together forms the total PDE loss:

* **Momentum Balance:**

    div(sigma) = 0

    (Forces must sum to zero for static equilibrium).

* **Continuity Equation:**

    div(u) - p / lambda = 0

    (Enforces the constitutive relationship between volumetric strain and pressure. As lambda approaches infinity, div(u) approaches 0).

b. Boundary Condition Enforcement

* **Dirichlet BC (L_dirichlet):**

    Enforces the "clamped" condition at the left vertical edge (x=0):

    u(0, y) = 0

    v(0, y) = 0

* **Neumann BC (L_neumann):**

    Enforces the vertical shear load at the right tip (x=48). The traction vector 't' is calculated as `t = sigma . n`:

    t_y = 0.0625 (Shear Load)

* **Traction-Free BC (L_free):**

    Enforces traction residual on the top and bottom slanted edges of the membrane.

#### 2. Neural Network Architecture
**The neural network architecture is defined in ```model.py```, with parameter settings in ```config.py```**

The network approximates the solution map `(x, y) -> (u, v, p)`.

* **Inputs:** Spatial coordinates x, y.
* **Outputs:** 3 scalar fields:
    * **u, v:** Displacement components in x and y.
    * **p:** Hydrostatic pressure (separate output to avoid volumetric locking).
* **Hidden Layers:** 4 fully connected layers. This is shallow enought to allow strong gradient propagation from tip to root.
* **Neurons:** 128 neurons per hidden layer. This wide network allows for better capture of the sharp gradient at the top-left corner due to shear loading.
* **Activation Function:** `tanh` (Hyperbolic Tangent). This function is smooth and infinitely differentiable, which is essential for computing the second-order derivatives required by the elasticity equations.
* **Normalization:** The input coordinates (x, y) are normalized before entering the network layers. I use a standard scaling approach `(x - mean) / scale` to keep input values roughly within the range [-1, 1], improving training stability.
* **Gradient Conflict Strategy:** I use a gradient surgery technique by PCGrad method to handle potential conflicts between different loss components. During backpropagation, the gradients for each loss term (PDE, Dirichlet, Neumann, Free) are computed individually. The dot product between the gradient vectors of different tasks are computed. If the dot product is negative, the gradient of one task is projected onto the normal plane of the other. The code for this is written in ```helpers.py```, and implemented in ```train.py```.

### Part C: Implementation and Results
#### 1. Implementation
The PINN framework is organized into modular Python files to separate concerns between configuration, physics, and training logic:

* **`config.py`**: Defines all hyperparameters via dataclasses, including material properties, geometry dimensions, network architecture (4 layers, 128 neurons), and the multi-stage training schedule. It allows for flexible "Loss Weighting" strategies to balance the competing objectives of momentum and incompressibility.
* **`model.py`**: Implements the Neural Network using TensorFlow/Keras. It features a custom `PINN` class that normalizes inputs to [-1, 1] and initializes weights using Variance Scaling to prevent vanishing gradients. The output layer predicts three variables: u, v, p.
* **`physics.py`**: Contains the core physics residuals. It implements the "Mixed Formulation" by explicitly calculating the Divergence of Stress for momentum and the Divergence of Displacement for continuity. It handles the specific constitutive relations for the nearly incompressible Cook's membrane.
* **`loss.py`**: Aggregates the individual loss components. It computes the Mean Squared Error (MSE) for the PDE residuals, Dirichlet boundary conditions (clamped edge), and Neumann boundary conditions (traction load), applying the weights defined in the configuration.
* **`sampling.py`**: Manages the generation of collocation points. It implements a "Stress Focus" strategy, which creates a hybrid dataset: some points are distributed uniformly, while some are densely clustered around the top-left corner (0, 44) to resolve the stress singularity, and also points focused near the boundaries.
* **`train.py`**: Orchestrates the training loop. It handles the Loss Normalization strategy by calculating the magnitude of gradients at the start of training and scaling them to ensure all physics terms contribute equally. It also manages the multi-stage training (first stage for pushing the tip displacement and second stage for fine-tuning.
* **`helpers.py`**: Provides utility functions, including random seed setting for reproducibility, tensor conversion, and PCGrad (Gradient Surgery) utilities if needed to handle conflicting gradients.
* **`evaluate.py`**: Responsible for post-processing. It evaluates the trained model on a dense grid, reconstructs the stress fields from the predicted displacements and pressure, calculates the tip displacement error, and generates visualization plots.
* **`main.py`**: The entry point that ties everything together. It initializes the configuration, builds the model, executes the training pipeline, and triggers the final evaluation.

#### 2. Visualization
To analyze the performance of the PINN, visualizations are generated using the evaluation module.

* **Collocation Point Cloud (`collocation_cloud.png`)**: This plot visualizes the spatial distribution of the training points generated by `sampling.py`.

<p align="center">
    <a href="https://github.com/kunlun-wu/cooks_membrane_tf">
        <img src="https://raw.githubusercontent.com/kunlun-wu/cooks_membrane_tf/main/figures/collocation_cloud.png">
    </a>
</p>

* **Evaluation Grid (`evaluation_grid.png`)**: Shows the structured mesh used for post-training inference. Unlike the unstructured training cloud, this is a generated grid of $121 \times 111$ points that maps the logical trapezoidal domain to a structured array.

<p align="center">
    <a href="https://github.com/kunlun-wu/cooks_membrane_tf">
        <img src="https://raw.githubusercontent.com/kunlun-wu/cooks_membrane_tf/main/figures/evaluation_grid.png">
    </a>
</p>

* **Training History (`training_history.png`)**: A plot of the individual loss components (PDE, Dirichlet, Neumann, Continuity) over the training epochs. This will be shown in the next section for results.

* **Deformed Shape (`deformed_shape.png`)**: This plot overlays the original undeformed trapezoid (dashed line) with the predicted deformed shape (solid line). This will be shown in the next section for results.

* **Von Mises Stress (`von_mises.png`)**: A contour plot of the Von Mises stress distribution within the membrane. This will be shown in the next section for results.

#### 3. Results (Comparison with Richardson Extrapolation of 16.43258437)

### Part D: Critical Reflections
#### 1. Limitations and Strengths
#### 2. Suggestions for Improvement
#### 3. What about elastoplastic materials?

### Part E: Transfer Learning
due to time constraints this is not finished.

## Section 2: Adaptive Sampling using Reinforcement Learning
**this part would be my final project, not finished yet.**
### Part A: Understanding Adaptive Sampling
#### 1. Motivation for Adaptive Sampling
#### 2. Overview of Reinforcement Learning

### Part B: RL-based Adaptive Sampling Strategy
#### 1. State, Action, Reward Design
#### 2. Integration with PINN Training Loop

### Part C: Implementation and Results
#### 1. Implementation
#### 2. Visualization
#### 3. Comparison with Uniform Sampling

### Part D: Critical Reflections
#### 1. Limitations and Strengths
#### 2. Suggestions for Improvement