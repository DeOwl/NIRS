# Quantum Machine Learning Research

This repository contains the python files and environment configuration files used to develop, train, and evaluate Variational Quantum Classifiers (VQCs) as part of our research on quantum machine learning.

---

## Overview

The project explores the implementation of quantum classifiers, including data encoding via feature maps, ansatz circuit design, training with classical optimizers, and performance evaluation. The files were all used as a part of the data collection and the experiment process.

---

## Repository Contents

- **Jupyter Notebooks and python files:** contain all the source code
- **`env.yml`:** An Anaconda environment configuration file specifying all dependencies required to run the notebooks.
- **paper.docm:** The resuting paper submitted as part of Springer 2025

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DeOwl/NIRS.git
cd NIRS
```

### 2. Create the Conda Environment

Using the provided `env.yml` file, create a new environment:

```bash
conda env create -f env.yml
```

Activate the environment:

```bash
conda activate qiskit
```

### 3. Launch Jupyter Notebooks / python files

---
