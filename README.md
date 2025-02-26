# Bayesian Methods for Optimization and Uncertainty Estimation

This repository contains implementations of Bayesian methods for optimization, uncertainty estimation, and decision-making in various domains, including drug candidate optimization, satellite image classification, and probabilistic artificial intelligence. The project demonstrates the use of Bayesian Optimization (BO), Bayesian Neural Networks (BNNs), and Gaussian Processes (GPs) to solve complex problems with constraints and uncertainty.

## Table of Contents
- [Overview](#overview)
- [Tasks](#tasks)
  - [Task 1: Gaussian Process Regression for Air Pollution Prediction](#task-1-bayesian-optimization-for-drug-candidate-effectiveness)
  - [Task 2: Bayesian Neural Networks for Satellite Image Classification](#task-2-bayesian-neural-networks-for-satellite-image-classification)
  - [Task 3: Bayesian Optimization for Drug Candidate Effectiveness](#task-3-probabilistic-artificial-intelligence)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview
This project explores the application of Bayesian methods to solve real-world problems involving optimization, uncertainty estimation, and decision-making. The tasks include:
1. **Bayesian Optimization** for maximizing drug candidate effectiveness while ensuring synthetic accessibility.
2. **Bayesian Neural Networks** for well-calibrated predictions in satellite image classification.
3. **Probabilistic Artificial Intelligence** for Bayesian hypothesis testing and uncertainty estimation.

Each task is implemented in Python, leveraging libraries such as `numpy`, `scipy`, `torch`, and `matplotlib`.

## Tasks

### Task 1: Gaussian Process Regression for Air Pollution Prediction
**Description**:  
This task involves using Bayesian Optimization to maximize the bioavailability (logP) of a drug candidate while ensuring its synthetic accessibility (SA) remains below a safety threshold. The algorithm iteratively evaluates structural features (x) to find the optimal solution that maximizes logP within the given constraints.

**Key Features**:
- Implements a custom Bayesian Optimization algorithm (`BOAlgorithm`).
- Balances exploration and exploitation using an acquisition function.
- Ensures all evaluated points satisfy the synthetic accessibility constraint.

**Files**:
- `solution.py`: Implementation of the Bayesian Optimization algorithm.
- Dummy functions (`f` and `v`) for testing and illustration.

---

### Task 2: Bayesian Neural Networks for Satellite Image Classification
**Description**:  
This task implements a Bayesian Neural Network (BNN) using Stochastic Weight Averaging-Gaussian (SWAG) for land-use classification of satellite images. The goal is to provide well-calibrated predictions and identify ambiguous images based on predicted confidence.

**Key Features**:
- Uses SWAG-Diagonal and SWAG-Full methods for approximate Bayesian inference.
- Provides uncertainty estimates for predictions.
- Evaluates model performance using accuracy, calibration, and cost metrics.

**Files**:
- `solution.py`: Implementation of the SWAG inference and model training.
- `util.py`: Utility functions for evaluation, calibration, and visualization.
- Dataset files (`train_xs.npz`, `train_ys.npz`, `val_xs.npz`, `val_ys.npz`).

---

### Task 3: Bayesian Optimization for Drug Candidate Effectiveness
**Description**:  
This task demonstrates Bayesian hypothesis testing using three probability distributions: Normal, Laplace, and Student-t. The goal is to compute the posterior probabilities of each hypothesis given observed data.

**Key Features**:
- Implements Bayesian hypothesis testing for probabilistic inference.
- Computes log-posterior probabilities for three hypotheses.
- Evaluates the model using synthetic data and posterior probabilities.

**Files**:
- `solution.py`: Implementation of Bayesian hypothesis testing.
- Dataset generation and evaluation functions.

---

## Installation
To run the code in this repository, you need Python 3 and the following libraries:
- `numpy`
- `scipy`
- `torch`
- `matplotlib`

You can install the required libraries using `pip`:
```bash
pip install numpy scipy torch matplotlib
