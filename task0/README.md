# Probabilistic Artificial Intelligence: Bayesian Hypothesis Testing

This project demonstrates Bayesian hypothesis testing using three probability distributions: Normal, Laplace, and Student-t. The goal is to compute the posterior probabilities of each hypothesis given observed data.

## Table of Contents

- [Introduction](#introduction)
- [Code Overview](#code-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

In Bayesian inference, we update our beliefs about hypotheses based on observed data. This project implements a simple Bayesian model with three hypotheses:

1. **Normal Distribution** (`norm`)
2. **Laplace Distribution** (`laplace`)
3. **Student-t Distribution** (`t`)

The prior probabilities for these hypotheses are given, and the posterior probabilities are computed using the log-likelihood of the data under each hypothesis.

## Code Overview

The project consists of the following components:

- **Hypothesis Space**: Three probability distributions with predefined parameters.
- **Prior Probabilities**: Probabilities assigned to each hypothesis before observing data.
- **Data Generation**: A function to generate synthetic data from one of the hypotheses.
- **Posterior Calculation**: Functions to compute the log-posterior and posterior probabilities of the hypotheses given the data.

### Key Functions

- `generate_sample(n_samples, seed=None)`: Generates synthetic data from one of the hypotheses.
- `log_posterior_probs(x)`: Computes the log-posterior probabilities for the three hypotheses.
- `posterior_probs(x)`: Converts log-posterior probabilities to posterior probabilities.
- `main()`: Demonstrates the functionality by generating data and computing posterior probabilities.

## Installation

To run this project, you need Python 3 and the following libraries:

- `numpy`
- `scipy`

You can install the required libraries using `pip`:

```bash
pip install numpy scipy
