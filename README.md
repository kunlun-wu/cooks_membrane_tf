# Cook's Membrane Problem with PINN and Adaptive Sampling using RL
The ReadMe file is written as the deliverable for course project in "Computational Solid Mechanics with AI." It is separated into two main sections: (1) Implementation of PINN for Cook's Membrane Problem (midterm project), and (2) Adaptive Sampling using Reinforcement Learning (final project). TensorFlow is used as the main framework for implementing the PINN and RL algorithms.

Cook's membrane problem is a 2D plane-stress/strain benchmark problem that involves simulating the deformation of a tapered cantilever under shear loading at the free end. The goal of this project is to solve the displacement field using PINN by enforcing equilibrium equations, boundary conditions, and constitutive laws within the loss function, and to enhance the training process through adaptive sampling strategies driven by RL. For this project, I refer to a linear elastic model with E = 1.0 and ν = 0.499999975.

## Section 1: Implementation of PINN for Cook's Membrane Problem
### Part A: Understanding the Physics
#### 1. Derivation of 2D Elasticity Equations (Navier Equations)
#### 2. Defining Linear Elastic Constitutive Relation (Hooke's Law)

### Part B: PINN Formulation
#### 1. Total Loss Function
#### 2. Neural Network Architecture

### Part C: Implementation and Results
#### 1. Implementation
#### 2. Visualization
#### 3. Comparison with Richardson Extrapolation (16.43258437)

### Part D: Critical Reflections
#### 1. Limitations and Strengths
#### 2. Suggestions for Improvement
#### 3. What about elastoplastic materials?

### Part E: Transfer Learning

## Section 2: Adaptive Sampling using Reinforcement Learning
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