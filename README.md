# Machine Learning from Scratch

Implementation of Machine Learning algorithms from scratch using only NumPy (without scikit-learn), with mathematical intuition and practical examples.

## Algorithms Implemented

### Supervised Learning

- Linear Regression (Normal Equation)
- Linear Regression (Gradient Descent)
- Logistic Regression
- K-Nearest Neighbours (KNN)
- Gaussian Naive Bayes
- Decision Tree Classifier (Gini Impurity)
- Random Forest Classifier

### Unsupervised Learning

- K-Means Clustering
- Principal Component Analysis
- DBSCAN(Density Based Spatial Clustering of Applications with Noise)

## Requirements

- Python 3.10+
- NumPy 

## Project Structure
```
machine-learning-from-scratch/
│
├── README.md 
├── requirements.txt
│
├── algorithms/
│   ├── supervised/
│   │   ├── linear_regression_closed.py
│   │   ├── linear_regression_gradient_descent.py
│   │   ├── logistic_regression.py
│   │   ├── k_nearest_neighbours.py
│   │   ├── gaussian_naive_bayes.py
│   │   ├── decision_tree_classifier.py
│   │   └── random_forest_classifier.py
│   │
│   └── unsupervised/
│   │   ├── k_means_clustering.py
│   │   ├── principal_component_analysis.py
│   │   ├── 
│
│
└── datasets/
```

## Goal

The goal of this repository is to understand the mathematics, intuition, and implementation details of Machine Learning algorithms by building them from scratch using only NumPy, without relying on machine learning libraries such as scikit-learn.
