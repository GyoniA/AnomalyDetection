# Anomaly Detection

This is my Masters Project Laboratory 2 project about anomaly detection using machine learning

# Comparison of different types of anomaly detection

## Statistics

Simpler statistics-based approaches work well for simpler well-defined datasets  
Some examples:
- Set static thresholds based on past data and mark exceeding values as anomalies
- Z-Score: count number of categories that deviate from their mean by a threshold and mark samples which exceed a set number of allowed deviations as anomalies
- Covariance Matrix: build covariance matrix from past data, values that deviate from expected covariance more than an allowed threshold are marked as anomalies

## Machine Learning

Use of machine learning is better for complex, large and variable datasets. The models tune their definition of anomalies based on a provided dataset

### Supervised Learning

Has access to labeled data of normal and anomalous examples
Some examples:
- K-nearest neighbors
- Linear/Polynomial regression
- Decision trees

### Unsupervised Learning

When there is no access to labeled data, models find their own relations to determine anomalies
Some examples:
- K means clustering
- Autoencoders
- Isolation forest
- Local Outlier Factor

### Semi-supervised Learning

Some labeled and unlabeled data, acts like supervised learning for labeled data, and tries to approximate for unlabeled data  

Usually less accurate than supervised learning, but can be useful in cases where only a small set of labeled data is available with a lot of unlabeled data and labeling is expensive

### Deep learning

Artificial neural networks with a large number of layers can be useful for identifying anomalies in complex datasets, where other methods could not work well enough


# Progress Report

## 1. Week

Looked into and added comparison of different types of anomaly detection techniques













## Sources

https://medium.com/@reza.shokrzad/6-pivotal-anomaly-detection-methods-from-foundations-to-2023s-best-practices-5f037b530ae6
