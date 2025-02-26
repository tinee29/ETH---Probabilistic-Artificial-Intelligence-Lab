# Gaussian Process Regression for Air Pollution Prediction

This project focuses on predicting PM2.5 pollution levels using Gaussian Process (GP) Regression. The goal is to model pollution concentrations across a city, with special attention to residential areas where underprediction carries a higher cost. The project involves data preprocessing, model training, and evaluation, with an emphasis on handling asymmetric cost functions and optimizing kernel selection.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Introduction

Air pollution prediction is critical for public health and urban planning. This project uses Gaussian Process Regression to predict PM2.5 levels across a city. The dataset consists of 2D coordinates and corresponding pollution levels, with additional flags indicating whether a location is in a residential area. The cost function is asymmetric, penalizing underprediction in residential areas more heavily.

## Project Structure

The project consists of the following components:

- **Data Loading**: Load training and test datasets (`train_x.csv`, `train_y.csv`, `test_x.csv`).
- **Feature Extraction**: Extract 2D coordinates and residential area flags from the data.
- **Model Training**: Train a Gaussian Process Regression model using a Matern-3/2 kernel.
- **Prediction**: Generate predictions for test data, incorporating a penalty for underprediction in residential areas.
- **Extended Evaluation**: Visualize predictions over a grid and save results.

### Key Files

- `solution.py`: Main script containing the implementation of the Gaussian Process Regression model.
- `train_x.csv`: Training data features.
- `train_y.csv`: Training data targets (pollution levels).
- `test_x.csv`: Test data features.

## Installation

To run this project, you need Python 3 and the following libraries:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

You can install the required libraries using `pip`:

```bash
pip install numpy scipy scikit-learn matplotlib
