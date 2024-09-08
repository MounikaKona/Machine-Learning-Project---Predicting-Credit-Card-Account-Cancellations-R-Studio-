
# Predicting Credit Card Account Cancellations

The Objective of the project is to explore the factors that lead to customers canceling their credit card accounts and develop machine learning algorithms that will predict the likelihood of a customer canceling their account in the future.
This repository contains code for training and evaluating models to predict customer status for credit card customers. We explore various models including logistic regression, decision trees, and random forests.



## Credit Card Account Data
Credit Card Account Data:
The credit_card_df dataframe contains information on the customers of a large US bank which provides a number of financial services including multiple credit card offerings.
The bank is looking to determine the factors that lead to customers canceling their credit card account and whether it can predict if a customer will cancel their account in the future.

Dataset Details:
The outcome variable in this data is customer_status, which indicates whether a customer eventually closed their account, resulting in a financial loss to the company.
The dataset contains a mixture of customer demographics and their financial behavior.

## Business Problem

To maintain profits, banks must maximize the number of customers with credit lines and encourage customers to carry large credit card balances from month-to-month to maximize revenue from interest charges. The bank has experienced record levels of customer account closures in recent years, leading to declining revenue. The bank aims to better identify customers at risk of canceling their account to minimize financial losses.

Key Questions:
What factors are associated with customers closing their credit card accounts?
Is it possible to predict whether a customer will close their account? If so, how accurate are the predictions?
How many costly errors is the model expected to produce?
Are there any actions or policies the company can implement to reduce the risk of losing customers?## Exploratory Data Analysis

The goal of this analysis is to explore the relationship between customer_status and other variables in the dataset. This analysis helps to discover which variables drive the differences between customers who do and do not close their account.

Questions Explored
What is the relationship between customer status and the number of dependents in the family?
Is there any relationship between customer status and whether customers were contacted by the company last year?
What is the relationship between employment status and customer status?
Is there a relationship between age, credit card limit, and customer status?
What is the relationship between card type, total spend amount, and customer status?
## Machine Learning Models
General Process
The credit_card_df data was split into training and test sets.
A feature engineering pipeline was defined using tidymodels::recipe that includes transformations for numeric and categorical features.
The recipe was prepared with the training data and then applied to the same training data.
Model 1: Logistic Regression
The parsnip package within the tidymodels framework was used to create, fit, and extract a logistic regression model.
Visualized the importance of features in the trained logistic regression model.
The yardstick package within the tidymodels framework was used to make predictions and evaluate the performance of the trained logistic regression model. Predictions were bound to the test dataset, evaluation metrics were computed, and the ROC curve was visualized.
Model 2: Decision Tree Classification
The rpart package was used to train a decision tree classifier.

![image](https://github.com/user-attachments/assets/95105e75-d8a8-4f23-9b3e-90182778139f)

The model was tuned using grid search to find the optimal hyperparameters.
Visualized the decision tree and the importance of features.
Model 3: Random Forest Classification
The ranger package was used to train a random forest classifier.
The model was tuned using random search to find the optimal hyperparameters.
Visualized the importance of features in the trained random forest model."# Machine-Learning-Project---Predicting-Credit-Card-Account-Cancellations-R-Studio-" 
