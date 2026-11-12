# Bank Marketing Classification Analysis

**Required Assignment 17.1: Comparing Classifiers**

UC Berkeley Machine Learning & Artificial Intelligence Program

## Project Overview

For this project, I used machine learning to predict whether a customer would subscribe to a bank term deposit. The dataset comes from a Portuguese bank's telephone marketing campaigns.

The goal was to compare different classification models and determine which one could help the bank make better marketing decisions.

I compared four models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

## Business Problem

The bank spends time and money contacting customers about term deposits, but most customers do not subscribe.

If the bank can identify customers who are more likely to subscribe, it could focus its marketing efforts on those customers instead of contacting everyone equally.

The main question I wanted to answer was:

**How accurately can machine learning models predict whether a customer will subscribe to a term deposit?**

## Dataset

The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

It originally contained 41,188 records and 21 columns.

For this analysis, I focused on seven customer characteristics:

- Age
- Job
- Marital status
- Education
- Credit default status
- Housing loan
- Personal loan

The target variable was whether the customer subscribed to a term deposit (`yes` or `no`).

I excluded call duration because that information would not be available before contacting a customer.

## Data Preparation

Before training the models, I cleaned and prepared the dataset.

I removed duplicate records, converted the target variable into numerical values, encoded categorical features, and standardized numerical features.

I divided the dataset into training and testing sets. To reduce training time, I used a stratified sample of 5,000 training records for model comparison.

## Model Evaluation

I compared the models using accuracy, precision, recall, F1 score, average precision, and ROC AUC.

I also used three-fold cross-validation and GridSearchCV to test different hyperparameters.

Because only approximately 11.27% of customers subscribed, the dataset was imbalanced. For this reason, I focused mainly on average precision rather than accuracy alone.

### Final Model Results

| Model | Accuracy | Precision | Recall | Average Precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 58.14% | 15.51% | 61.03% | 0.1925 |
| KNN | 88.73% | 0.00% | 0.00% | 0.2031 |
| Decision Tree | 88.61% | 44.00% | 3.79% | 0.1871 |
| SVM | 88.73% | 0.00% | 0.00% | 0.1657 |

Logistic Regression achieved the highest average precision during cross-validation and was selected as the final model.

Although KNN had a slightly higher average precision on the test set, it did not identify any subscribers using its default classification threshold.

Logistic Regression identified approximately 61% of the customers who actually subscribed, although it also produced many false positives.

## Key Findings

One of the biggest findings was that accuracy alone was not a good way to evaluate these models.

A baseline model that always predicted customers would not subscribe achieved approximately 88.73% accuracy, but it failed to identify any actual subscribers.

The exploratory analysis also showed differences in subscription rates between customer groups.

Students had the highest observed subscription rate among job categories at approximately 31.43%, while blue-collar workers had the lowest at approximately 6.90%.

These differences show that customer characteristics may be useful for predicting subscriptions, although they do not necessarily indicate causation.

## Business Recommendations

Based on the results, I would recommend that the bank continue testing Logistic Regression as a tool for improving its marketing campaigns.

The bank could use the model to rank customers based on their predicted likelihood of subscribing and prioritize customers with higher scores.

However, the model still produces false positives, so the bank should consider the cost of contacting customers before implementing it.

I would recommend testing the model in a smaller marketing campaign and comparing its results against the bank's existing approach.

## Next Steps

To improve this project, I would consider training the models on the full dataset instead of the smaller sample.

I would also explore additional customer information that would be available before a phone call, test different classification thresholds, and evaluate whether the model improves marketing results in practice.

Another important step would be monitoring model performance over time because customer behavior and economic conditions can change.

## Conclusion

This project helped me better understand how different classification models perform on an imbalanced dataset.

Logistic Regression was selected based on cross-validation and provided a useful balance between identifying potential subscribers and making interpretable predictions.

The results suggest that machine learning could help the bank make more informed marketing decisions, although further testing would be needed before using the model in real campaigns.

## Jupyter Notebook

[View the complete analysis and Python code](practical_application_III.ipynb)

## Tools and Libraries

Python, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.
