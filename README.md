# Anomaly Detection

This is my Masters Project Laboratory 2 project about anomaly detection using machine learning

# Comparison of different types of anomaly detection

## Statistics

Statistics-based approaches work well for simpler well-defined datasets, where assumptions can be made about the data distribution (or the distribution of the data is known)  
Some examples:
- Percentile-based detection/Set static thresholds based on past data, where the model marks exceeding values as anomalies (most basic approach)
- Z-Score: count number of categories that deviate from their mean by a threshold and mark samples which exceed a set number of allowed deviations as anomalies
- Covariance Matrix: build covariance matrix from past data, values that deviate from expected covariance more than an allowed threshold are marked as anomalies
- Grubbs Test: used with normal distributions, where the model marks samples that deviate from the normal distribution more than a threshold as outliers

Advantages:
- Easy to implement
- Works well when data follows a known distribution

Disadvantages:
- Requires assumptions about the data distribution
- Doesn't work well for non-linear/high-dimensional data

Used mostly by financial institutions or healthcare providers

## Machine Learning

Use of machine learning is better for complex, large and variable datasets. The models tune their definition of anomalies based on a provided dataset

### Supervised Learning

Has access to labeled data of normal and anomalous examples  
Some examples:
- K-nearest neighbors
- Linear/Polynomial regression
- Decision trees
- Random Forests
- Support Vector Machines
- Gradient Boosting
- Neural Networks

Advantages:
- Can be used for a wide range of datasets
- Can learn complex relationships between features
- Performs well when provided with large enough labeled data

Disadvantages:
- Requires labeled data, which may be expensive to obtain
- Can struggle with rare anomalies that were not seen during training

Used when labeled data is available, for example transaction fraud detection

### Unsupervised Learning

When there is no access to labeled data, models find their own relations to determine anomalies  
Some examples:
- K means clustering
- DBSCAN (Density-Based Spatial Clustering)
- Isolation forest
- Local Outlier Factor
- Autoencoders (deep learning)

Advantages:
- Doesn't require labeled data
- Can be used for a wide range of datasets
- Can learn complex relationships between features
- Can detect novel anomalies

Disadvantages:
- Usually not as accurate as supervised learning
- Can be harder to train

Used when labeled data is not available, for example network anomaly detection

### Semi-supervised Learning

Some labeled and unlabeled data, acts like supervised learning for labeled data, and tries to approximate for unlabeled data  

Usually less accurate than supervised learning, but can be useful in cases where only a small set of labeled data is available with a lot of unlabeled data and labeling is expensive

### Deep learning

Artificial neural networks with a large number of layers can be useful for identifying anomalies in complex datasets, where other methods could not work well enough  
Some examples:
- Autoencoders
- Generative Adversarial Networks

Advantages:
- Can learn complex/high-dimensional relationships between features
- Can detect novel anomalies
- Can be used for a wide range of datasets that can be very large

Disadvantages:
- Can be harder to train and interpret
- Can be computationally expensive
- Requires a lot of data to train

Used for complex/high-dimensional datasets, for example when working with images

## Time Series based

Time series based approaches focus on patters over time, where anomalies are defined as deviations from the normal patterns, such as spikes or drops in the data  
Some examples:
- Autoregressive Integrated Moving Average (ARIMA)
- Exponential Smoothing (ETS)
- LSTM (Long Short Term Memory) based models

Advantages:
- Effective at detecting seasonal or cyclical patterns

Disadvantages:
- Sensitive to seasonal patterns

Used for time series data, for example power consumption or network traffic

## Graph based

Graph based methods detect anomalies by analyzing the relationships and interactions between entities, often modeled as a graph  
Some examples:
- Graph Convolutional Networks (GCN)
- PageRank
- Random Walks

Advantages:
- Can detect complex relationships between entities
- Can capture both local and global patterns

Disadvantages:
- Can be computationally expensive
- Requires well structured data

Used by social media platforms, for example to detect fake accounts or suspicious behavior

---

# Results of comparison

## On labeled dataset

Used dataset: https://huggingface.co/datasets/pointe77/credit-card-transaction/tree/main

### Isolation Forest
Train time: ~6 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.97   | 0.98     | 553574  |
| 1.0          | 0.02      | 0.21   | 0.04     | 2145    |
| accuracy     |           |        | 0.96     | 555719  |
| macro avg    | 0.51      | 0.59   | 0.51     | 555719  |
| weighted avg | 0.99      | 0.96   | 0.98     | 555719  |

![covariance matrix](images/pointe77/IFcm.png)

---

### Local Outlier Factor
Train time: ~1202 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.63   | 0.77     | 553574  |
| 1.0          | 0.00      | 0.24   | 0.00     | 2145    |
| accuracy     |           |        | 0.63     | 555719  |
| macro avg    | 0.50      | 0.44   | 0.39     | 555719  |
| weighted avg | 0.99      | 0.63   | 0.77     | 555719  |

![covariance matrix](images/pointe77/LOFcm.png)

---

### One-Class SVM
Train time: ~2067 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.85   | 0.92     | 553574  |
| 1.0          | 0.01      | 0.32   | 0.02     | 2145    |
| accuracy     |           |        | 0.85     | 555719  |
| macro avg    | 0.50      | 0.58   | 0.47     | 555719  |
| weighted avg | 0.99      | 0.85   | 0.91     | 555719  |

![covariance matrix](images/pointe77/OCSVMcm.png)

---

### K-Means
Train time: ~0 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.54   | 0.70     | 553574  |
| 1.0          | 0.00      | 0.46   | 0.01     | 2145    |
| accuracy     |           |        | 0.54     | 555719  |
| macro avg    | 0.50      | 0.50   | 0.35     | 555719  |
| weighted avg | 0.99      | 0.54   | 0.70     | 555719  |

![covariance matrix](images/pointe77/KMcm.png)

---

### Autoencoder

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.98   | 0.99     | 553574  |
| 1.0          | 0.04      | 0.19   | 0.06     | 2145    |
| accuracy     |           |        | 0.98     | 555719  |
| macro avg    | 0.52      | 0.59   | 0.53     | 555719  |
| weighted avg | 0.99      | 0.98   | 0.99     | 555719  |

![covariance matrix](images/pointe77/AEcm.png)

---

### PR + ROC Curves

![PR+Roc.png](images/pointe77/PRAndRoc.png)

## On anonymized credit card fraud dataset

Used dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data  
There are 2 versions, one with SMOTE-ENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors) and one without.  
The confusion matrices are for the default model, which is the one without SMOTE-ENN. Confusion matrices for the SMOTE-ENN model are in the 'images' folder.

### Isolation Forest

Default Train time: ~1 second

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.98   | 0.99     | 71108   |
| 1.0          | 0.04      | 0.60   | 0.07     | 94      |
| accuracy     |           |        | 0.98     | 71202   |
| macro avg    | 0.52      | 0.79   | 0.53     | 71202   |
| weighted avg | 1.00      | 0.98   | 0.99     | 71202   |

SMOTE-ENN Train time: ~0 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.96   | 0.98     | 71108   |
| 1.0          | 0.03      | 0.77   | 0.05     | 94      |
| accuracy     |           |        | 0.96     | 71202   |
| macro avg    | 0.51      | 0.86   | 0.52     | 71202   |
| weighted avg | 1.00      | 0.96   | 0.98     | 71202   |


![covariance matrix](images/mlg-ulb/default/IFcm.png)

---

### Local Outlier Factor

Train time: ~42 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.77   | 0.87     | 71108   |
| 1.0          | 0.00      | 0.14   | 0.00     | 94      |
| accuracy     |           |        | 0.77     | 71202   |
| macro avg    | 0.50      | 0.45   | 0.44     | 71202   |
| weighted avg | 1.00      | 0.77   | 0.87     | 71202   |

SMOTE-ENN Train time: ~34 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.76   | 0.86     | 71108   |
| 1.0          | 0.00      | 0.39   | 0.00     | 94      |
| accuracy     |           |        | 0.76     | 71202   |
| macro avg    | 0.50      | 0.58   | 0.43     | 71202   |
| weighted avg | 1.00      | 0.76   | 0.86     | 71202   |

![covariance matrix](images/mlg-ulb/default/LOFcm.png)

---

### One-Class SVM

Train time: ~97 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.98   | 0.99     | 71108   |
| 1.0          | 0.04      | 0.65   | 0.08     | 94      |
| accuracy     |           |        | 0.98     | 71202   |
| macro avg    | 0.52      | 0.81   | 0.54     | 71202   |
| weighted avg | 1.00      | 0.98   | 0.99     | 71202   |

SMOTE-ENN Train time: ~73 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.98   | 0.99     | 71108   |
| 1.0          | 0.05      | 0.70   | 0.10     | 94      |
| accuracy     |           |        | 0.98     | 71202   |
| macro avg    | 0.53      | 0.84   | 0.54     | 71202   |
| weighted avg | 1.00      | 0.98   | 0.99     | 71202   |

![covariance matrix](images/mlg-ulb/default/OCSVMcm.png)

---

### K-Means

Train time: ~0 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.94   | 0.97     | 71108   |
| 1.0          | 0.00      | 0.17   | 0.01     | 94      |
| accuracy     |           |        | 0.94     | 71202   |
| macro avg    | 0.50      | 0.56   | 0.49     | 71202   |
| weighted avg | 1.00      | 0.94   | 0.97     | 71202   |

SMOTE-ENN Train time: ~0 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.06   | 0.11     | 71108   |
| 1.0          | 0.00      | 1.00   | 0.00     | 94      |
| accuracy     |           |        | 0.06     | 71202   |
| macro avg    | 0.50      | 0.53   | 0.06     | 71202   |
| weighted avg | 1.00      | 0.06   | 0.11     | 71202   |

![covariance matrix](images/mlg-ulb/default/KMcm.png)

---

### Autoencoder

Train time: ~65 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.98   | 0.99     | 71108   |
| 1.0          | 0.05      | 0.70   | 0.09     | 94      |
| accuracy     |           |        | 0.98     | 71202   |
| macro avg    | 0.52      | 0.84   | 0.54     | 71202   |
| weighted avg | 1.00      | 0.98   | 0.99     | 71202   |

SMOTE-ENN Train time: ~54 seconds

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.95   | 0.97     | 71108   |
| 1.0          | 0.02      | 0.81   | 0.04     | 94      |
| accuracy     |           |        | 0.95     | 71202   |
| macro avg    | 0.51      | 0.88   | 0.51     | 71202   |
| weighted avg | 1.00      | 0.95   | 0.97     | 71202   |

![covariance matrix](images/mlg-ulb/default/AEcm.png)

---

### Transformer

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.99   | 1.00     | 71108   |
| 1.0          | 0.10      | 0.68   | 0.18     | 94      |
| accuracy     |           |        | 0.99     | 71202   |
| macro avg    | 0.55      | 0.84   | 0.59     | 71202   |
| weighted avg | 1.00      | 0.99   | 0.99     | 71202   |

SMOTE-ENN:

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0.0          | 1.00      | 0.99   | 1.00     | 71108   |
| 1.0          | 0.10      | 0.76   | 0.18     | 94      |
| accuracy     |           |        | 0.99     | 71202   |
| macro avg    | 0.55      | 0.87   | 0.59     | 71202   |
| weighted avg | 1.00      | 0.99   | 0.99     | 71202   |

![covariance matrix](images/mlg-ulb/default/Transformercm.png)

### PR + ROC Curves

![PR+Roc.png](images/mlg-ulb/default/PRAndRoc.png)

SMOTE-ENN:

![PR+Roc.png](images/mlg-ulb/PRAndRoc.png)

---

## 1. Week

### Progress

Looked into and added comparison of different types of anomaly detection techniques  
Found potential datasets, that can be used for the project:
- https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset
- https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset  

Credit card fraud datasets:
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction
- https://data.world/vlad/credit-card-fraud-detection  

Network anomaly detection datasets:
- https://www.kaggle.com/datasets/kaiser14/network-anomaly-dataset?select=network_dataset_labeled.csv
- https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019

### Next week's goals

- Select used dataset
- Find a base architecture for the model
- Test dataset with simpler methods for comparison

---

## 2. Week

### Progress

Found more credit card fraud datasets, this one has labeled columns:
- https://huggingface.co/datasets/pointe77/credit-card-transaction/tree/main

And this one has a lot of recent data, but no labeled columns:
- https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

Added a comparison of the results of 4 different models on the labeled column credit card transaction dataset, I used Isolation Forest, Local Outlier Factor, One-Class SVM and K-Means clustering.

### Next week's goals

- Improve model parameters
- Explain the results
- Add an initial Deep Learning model and compare it with the others
- Maybe try another dataset
- Separate data reading and model training into separate files

---

## 3. Week

### Progress

Added model saving and loading, extracted the data loading and pre-processing into a separate file.  
Added an Autoencoder model and compared it with the other models.

### Next week's goals

- Improve model parameters
- Add short explanation of comparison metrics
- Improve Autoencoder architecture or try other deep learning models
- Extract paths to a config file

---

## 4. Week

### Progress

Added area under the precision-recall curve (AUPRC) metric to the PRC comparison plot, as this works better for imbalanced datasets.[[1]](#1)  
Looked for datasets and models in other papers about anomaly detection.  
Added data loading and comparison for the "Credit Card Fraud Detection" dataset from mlg-ulb, as this dataset is used as a benchmark in a lot of anomaly detection papers.  
Added SMOTE-ENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors) to the new dataset's loading step, because it improves the accuracy of the models according to this paper.[[2]](#2)


### Next week's goals

- Look into data simulation
- Look for dataset other than credit card fraud detection that is not anonymized?
- Add deep learning models other than autoencoders

---

## 5. Week

### Progress

Looked into which deep learning models work best for anomaly detection, and found that Transformer models work really well for tabular data (like the credit card fraud dataset).[[3]](#3)  
Added a Transformer model to the project and compared it with the other models.  
Looked for data generation methods that can be used to generate synthetic data for anomaly detection.
Potential data generation methods:
- Synthetic Minority Over-sampling Technique (SMOTE) (Like I tried previously)
- CTGAN (Conditional Tabular Generative Adversarial Networks) [[4]](#4), implemented in the SDV library [[5]](#5). Could be used for expanding small but labeled datasets
- Faker library with extra providers [[6]](#6) could be used to turn anonymized data into fake data

### Next week's goals

- Tune parameters of models (especially Transformer)
- Use data generation methods to improve model performance

## 6. Week

### Progress


### Next week's goals



---

## Sources / References
- https://medium.com/@reza.shokrzad/6-pivotal-anomaly-detection-methods-from-foundations-to-2023s-best-practices-5f037b530ae6
- https://arxiv.org/abs/1901.03407
- https://www.sciencedirect.com/science/article/abs/pii/S1084804515002891
- https://link.springer.com/article/10.1007/s40747-024-01446-8
- <a name="1">[1]</a> J. Hancock, T. M. Khoshgoftaar and J. M. Johnson, "Informative Evaluation Metrics for Highly Imbalanced Big Data Classification," 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA), Nassau, Bahamas, 2022, pp. 1419-1426, doi: 10.1109/ICMLA55696.2022.00224. keywords: {Measurement;Insurance;Machine learning;Receivers;Big Data;Data models;Robustness;Extremely Randomized Trees;XGBoost;Class Imbalance;Big Data;Undersampling;AUC;AUPRC}  
- <a name="2">[2]</a> https://ieeexplore.ieee.org/abstract/document/9698195
- <a name="3">[3]</a> https://arxiv.org/html/2406.03733v1#S4
- <a name="4">[4]</a> https://github.com/sdv-dev/CTGAN
- <a name="5">[5]</a> https://github.com/sdv-dev/SDV
- <a name="6">[6]</a> https://github.com/joke2k/faker
- Paper about data generation: https://ieeexplore.ieee.org/document/10072179

#### List of some papers that use the "Credit Card Fraud Detection" dataset from mlg-ulb:
[comment]: <> (TODO: Add better citations, change them to actual references)
- https://www.mdpi.com/2079-9292/11/4/662
- https://www.researchgate.net/profile/Dr-Kumar-Lilhore/publication/341932015_An_Efficient_Credit_Card_Fraud_Detection_Model_Based_on_Machine_Learning_Methods/links/5ee4a477458515814a5b891e/An-Efficient-Credit-Card-Fraud-Detection-Model-Based-on-Machine-Learning-Methods.pdf
- https://ieeexplore.ieee.org/abstract/document/8979331
- https://ieeexplore.ieee.org/abstract/document/9651991
- https://ieeexplore.ieee.org/abstract/document/9121114
