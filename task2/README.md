# Bayesian Neural Networks for Satellite Image Classification

This project implements a Bayesian Neural Network (BNN) using Stochastic Weight Averaging-Gaussian (SWAG) for land-use classification of satellite images. The goal is to provide well-calibrated predictions and identify ambiguous images based on predicted confidence. The model is trained using SWAG-Diagonal and SWAG-Full methods, which approximate the posterior distribution of the network weights.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Introduction

Satellite image classification is a critical task in land-use analysis, but it often involves ambiguous or uncertain data. This project uses Bayesian Neural Networks (BNNs) to provide well-calibrated predictions and quantify uncertainty. The model is trained using Stochastic Weight Averaging-Gaussian (SWAG), which approximates the posterior distribution of the network weights. This allows the model to not only classify images but also identify ambiguous samples.

## Project Structure

The project consists of the following components:

- **Data Loading**: Load training and validation datasets (`train_xs.npz`, `train_ys.npz`, `val_xs.npz`, `val_ys.npz`).
- **Model Training**: Train a Convolutional Neural Network (CNN) using SWAG-Diagonal or SWAG-Full.
- **Prediction**: Generate predictions for validation data, including uncertainty estimates.
- **Evaluation**: Evaluate the model's accuracy, calibration, and cost on the validation set.

### Key Files

- `solution.py`: Main script containing the implementation of the SWAG inference and model training.
- `util.py`: Utility functions for evaluation, calibration, and visualization.
- `train_xs.npz`, `train_ys.npz`: Training data features and labels.
- `val_xs.npz`, `val_ys.npz`: Validation data features and labels.

## Installation

To run this project, you need Python 3 and the following libraries:

- `numpy`
- `torch`
- `scipy`
- `matplotlib`

You can install the required libraries using `pip`:

```bash
pip install numpy torch scipy matplotlib
