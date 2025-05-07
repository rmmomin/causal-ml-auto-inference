
# Causal ML Auto-Inference

This repository provides a PyTorch implementation of the framework proposed in the paper:  
**[Deep Learning for Individual Heterogeneity: An Automatic Inference Framework](https://arxiv.org/abs/2010.14694)**  
by Max H. Farrell, Tengyuan Liang, and Sanjog Misra.

## Overview

The framework introduces a deep learning approach to automatically infer individual-level heterogeneity in treatment effects. It combines machine learning techniques with causal inference principles to estimate Conditional Average Treatment Effects (CATE). Illustrates estimation and inference for the estimator for the average treatment effect.

## Features

- **PyTorch Implementation**: Leverages PyTorch for building and training neural networks with a MSE loss function as an example.
- **Automatic Inference**: Uses automatic differentiation to compute gradients for the influence function estimator.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.7 or higher
- NumPy
- Pandas
- Scikit-learn

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/rmmomin/causal-ml-auto-inference.git
cd causal-ml-auto-inference
```

### Usage

The main scripts and modules are located in the `code/` directory.

## Directory Structure

```
causal-ml-auto-inference/
├── code/                   # Source code for model training and evaluation
├── outputs/                # Saved models and results
└── README.md               # Project documentation
```

## References

- **Original Paper**: [Deep Learning for Individual Heterogeneity: An Automatic Inference Framework](https://arxiv.org/abs/2010.14694)
- **Original R Code**: [R Implementation](https://github.com/MisraLab/cml.github.io/tree/main/Lecture%208)
- [Another, more flexible PyTorch implementation](https://github.com/connachermurphy/causal-machine-learning) 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is developed by [Rayhan Momin](https://github.com/rmmomin), a Finance PhD student at UChicago Booth.

---

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/rmmomin/causal-ml-auto-inference/issues).
