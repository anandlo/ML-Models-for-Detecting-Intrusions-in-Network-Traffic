# Intrusion Detection System Using Machine Learning Algorithms

## Problem Statement

The task is to build a network intrusion detector — a predictive model capable of distinguishing between bad connections, called intrusions or attacks, and good, normal connections.

## Introduction

An **Intrusion Detection System (IDS)** is a software application designed to detect network intrusions using various machine learning algorithms. The IDS monitors a network or system for malicious activity and protects a computer network from unauthorized access, potentially including insider threats. The main goal of this project is to build a predictive model (i.e., a classifier) capable of distinguishing between "bad connections" (intrusions/attacks) and "good (normal) connections".

Attacks can be categorized into four main types:

1. **DOS (Denial-of-Service)**: Examples include SYN flood attacks, which aim to make a service unavailable.
2. **R2L (Remote-to-Local)**: Unauthorized access from a remote machine, such as password guessing.
3. **U2R (User-to-Root)**: Unauthorized access to local superuser (root) privileges, such as buffer overflow attacks.
4. **Probing**: Surveillance or other types of reconnaissance, like port scanning.

## Dataset Used: KDD Cup 1999 Dataset

### Dataset Description

The dataset used for this project is the **KDD Cup 1999 dataset**, which is widely used for evaluating intrusion detection systems. The dataset consists of the following data files:

- **kddcup.names**: A list of features.
- **kddcup.data.gz**: The full data set.
- **kddcup.data_10_percent.gz**: A 10% subset of the full data set.
- **kddcup.newtestdata_10_percent_unlabeled.gz**: Unlabeled test data (10% subset).
- **kddcup.testdata.unlabeled.gz**: Unlabeled full test data.
- **kddcup.testdata.unlabeled_10_percent.gz**: Unlabeled 10% subset of the test data.
- **corrected.gz**: Test data with corrected labels.
- **training_attack_types**: A list of intrusion types.
- **typo-correction.txt**: A brief note on a typo in the dataset that has been corrected.

## Machine Learning Algorithms Applied

Various machine learning algorithms have been applied to develop the intrusion detection system, including:

- **Gaussian Naive Bayes**
- **Decision Trees**
- **Random Forest**
- **Support Vector Machine (SVC)**
- **Logistic Regression**
- **Gradient Descent**

## Model Performance Metrics

The performance of each model was evaluated based on **train scores**, **test scores**, and the **time taken for training and testing**. Below are the results for each of the models:

### Accuracy Scores

| Model               | Train Score | Test Score  |
|---------------------|-------------|-------------|
| Gaussian NB         | 0.8795      | 0.8790      |
| Decision Trees      | 0.9906      | 0.9905      |
| Random Forest       | 0.99997     | 0.99966     |
| SVC                 | 0.9987      | 0.9988      |
| Logistic Regression | 0.9935      | 0.9935      |
| Gradient Descent    | 0.9979      | 0.9977      |

### Training and Testing Times

| Model               | Training Time (s) | Testing Time (s) |
|---------------------|-------------------|------------------|
| Gaussian NB         | 0.652             | 0.674            |
| Decision Trees      | 0.945             | 0.161            |
| Random Forest       | 10.125            | 1.132            |
| SVC                 | 400.425           | 105.411          |
| Logistic Regression | 6.805             | 0.068            |
| Gradient Descent    | 348.187           | 2.380            |

## Visualizations

To better understand the performance of each model, visualizations were created to illustrate the differences in accuracy and computation times. These include:

1. **Accuracy Comparison**: A bar chart comparing the train and test scores for each model to highlight subtle differences in accuracy.
2. **Training and Testing Times**: A horizontal bar chart that shows the training and testing times, with a logarithmic scale used to emphasize significant time differences between models.

## Conclusion

This project provides a comprehensive overview of how different machine learning algorithms can be used to build an intrusion detection system. The results show that **Random Forest** and **SVC** provide the highest accuracy, while **Gaussian Naive Bayes** is the fastest to train and test. Choosing the best model depends on the trade-off between accuracy and computation time for the specific use case.

## Future Work

- **Hyperparameter Tuning**: Perform hyperparameter tuning to improve the accuracy of each model.
- **Feature Engineering**: Explore additional feature engineering techniques to improve model performance.
- **Deep Learning Models**: Experiment with deep learning models to see if they can outperform traditional machine learning models.

## License

This project is licensed under the MIT License.

## Contact

For questions or contributions, please reach out to:

- **Author**: Anand Lo
- **Email**: anandlo@dal.ca
- **LinkdIn**: https://www.linkedin.com/in/anandlo/

Feel free to contribute to the project or raise issues for any suggestions or improvements!

