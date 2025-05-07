
# Causal ML Auto-Inference

This repository provides a PyTorch implementation of the framework proposed in the paper:  
**[Deep Learning for Individual Heterogeneity: An Automatic Inference Framework](https://arxiv.org/abs/2010.14694)**  
by Max H. Farrell, Tengyuan Liang, and Sanjog Misra.

## Overview

The framework introduces a deep learning approach to automatically infer individual-level heterogeneity in treatment effects. It combines machine learning techniques with causal inference principles to estimate Conditional Average Treatment Effects (CATE) without manual feature engineering.

## Features

- **PyTorch Implementation**: Leverages PyTorch for building and training neural networks tailored for causal inference tasks.
- **Automatic Inference**: Automates the process of estimating individual treatment effects using deep learning.
- **Reproducibility**: Includes code to replicate experiments and results from the original paper.

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

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Usage

The main scripts and modules are located in the `code/` directory.

To train the model:

```bash
python code/train_model.py --config configs/default.yaml
```

To evaluate the model:

```bash
python code/evaluate_model.py --model_path outputs/model.pth
```

Replace `configs/default.yaml` and `outputs/model.pth` with your configuration file and trained model path, respectively.

## Directory Structure

```
causal-ml-auto-inference/
├── code/                   # Source code for model training and evaluation
├── configs/                # Configuration files for experiments
├── data/                   # Dataset files
├── outputs/                # Saved models and results
├── requirements.txt        # Python dependencies
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
