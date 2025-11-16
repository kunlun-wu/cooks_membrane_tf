# Cook's Membrane Problem with PINN and Adaptive Sampling using RL
The ReadMe file is written as the deliverable for course project in "Computational Solid Mechanics with AI." It is separated into two main sections: (1) Implementation of PINN for Cook's Membrane Problem (midterm project), and (2) Adaptive Sampling using Reinforcement Learning (final project). TensorFlow is used as the main framework for implementing the PINN and RL algorithms.

Cook's membrane problem is a 2D plane-stress/strain benchmark problem that involves simulating the deformation of a tapered cantilever under shear loading at the free end. The goal of this project is to solve the displacement field using PINN by enforcing equilibrium equations, boundary conditions, and constitutive laws within the loss function, and to enhance the training process through adaptive sampling strategies driven by RL. For this project, I refer to a linear elastic model with E = 1.0 and ν = 0.499999975.

## Section 1: Implementation of PINN for Cook's Membrane Problem
### Part A: Understanding the Physics
#### Derivation of 2D Elasticity Equations (Navier Equations)
We work in 2D with coordinates \((x,y)\). The unknown is the displacement field  

\[
\mathbf{u}(x,y) = 
\begin{bmatrix}
u_x(x,y)\\
u_y(x,y)
\end{bmatrix}.
\]

Assumptions:

- Small deformations (small-strain kinematics).
- Homogeneous, isotropic, linear elastic material.
- Static equilibrium with no body forces.
- Plane stress or plane strain reduction to 2D.

The small strain tensor is
\[
\boldsymbol{\varepsilon} 
= \frac{1}{2}\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^T\right).
\]

In components:
\[
\varepsilon_{xx} = \frac{\partial u_x}{\partial x},\quad
\varepsilon_{yy} = \frac{\partial u_y}{\partial y},\quad
\varepsilon_{xy} = \frac{1}{2}\left(
\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x}
\right).
\]

#### Defining Linear Elastic Constitutive Relation (Hooke's Law)