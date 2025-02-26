# Bayesian Optimization for Drug Candidate Effectiveness

This project implements a Bayesian Optimization (BO) algorithm to maximize the bioavailability (logP) of a drug candidate while ensuring its synthetic accessibility (SA) remains below a safety threshold. The algorithm iteratively evaluates structural features (x) to find the optimal solution that maximizes logP within the given constraints.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Introduction

The goal of this project is to optimize the structural features of a drug candidate to maximize its bioavailability (logP) while ensuring that the synthetic accessibility (SA) remains below a safety threshold. Bayesian Optimization is used to iteratively evaluate and refine the structural features, balancing exploration and exploitation to find the optimal solution.

## Project Structure

The project consists of the following components:

- **BOAlgorithm Class**: Implements the Bayesian Optimization algorithm, including acquisition function optimization, observation handling, and recommendation of the next point to evaluate.
- **Objective and Constraint Functions**: Dummy functions (`f` and `v`) simulate the logP and SA evaluations, respectively.
- **Main Function**: Demonstrates the usage of the BOAlgorithm class with a toy problem.

### Key Files

- `solution.py`: Main script containing the implementation of the Bayesian Optimization algorithm.
- **Dummy Functions**: `f` (logP objective) and `v` (SA constraint) are provided for testing and illustration.

## Installation

To run this project, you need Python 3 and the following libraries:

- `numpy`
- `scipy`

You can install the required libraries using `pip`:

```bash
pip install numpy scipy
