
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
## Exploratory Data Analysis
To explore the relationship between customer_status and the other variable in the credit_card_df, the goal is to discover which variables drive the differences between customers who do and do not close their account.

What is the relationship between customer status and the number of dependents in the family?

![image](https://github.com/user-attachments/assets/d2e86763-1202-4c65-8e87-2569ea0a9340)

From the analysis, it is evident that, irrespective of the number of dependents the number of active accounts is always greater than the number of closed accounts. Moerover  there are more customers with 2 or 3 dependents. The number of closed and active accounts increased as the number of dependents in the family increases until 3, latter the number of closed and active accounts descreased as the number of depedents increased morethan 4 drastically. There are very few customers who have 5 depedents.

Is there any relationship between customer status and the customers who were contacted by the company last year and those who are not?

![image](https://github.com/user-attachments/assets/39ab4549-e80e-4c51-a8a1-c741fa20a074)

From this analysis it is evident that there is some relation between the customer status, total amount spent last year and the number of times the sales representative contacted them. the closed accounts customers have spent less amount in the last one year when compared to the amount spent by the customers who have active acounts. From this analysis we can undertand that if any new customer is spending less than the average of 4500 there is a possibility that customer tend to close the account. similarly at that point the sales representative can contact the customer to offer any discounts etc. From the left graph it is evident that the average spending amount is greater for the closed account who were contacted 2 times by the sales representative. On the right graph it is evident that if the company contact morethan 1 time the spendig amount is decreased.

What is the relationship between employment status and customer status?

![image](https://github.com/user-attachments/assets/27e18fca-9189-4e93-bb07-67d4e7c8a20c)

From this analysis, we can understand that employment status has an impact on the account status. Customers who have full time job tend to stay with the credit card company when compared to partime job holders and self employed customers. Similarly, there are more number of closed acounts related to part time job holding customers which mean there is higher tendency for any new customer with partitme job to close the acount.

Is there a relationship between age, credit card limit, and customer status?

![image](https://github.com/user-attachments/assets/a2880f98-8d40-487b-bc78-b9a8f20d8b29)

![image](https://github.com/user-attachments/assets/1967cb32-7f40-4937-8801-548a2dbedf86)


Based on the analysis it is evident that the age group between 38 and 48 have the highest mean credit card limit and the most active accounts. The next best age group is 48 to 58 with second highest average credit card limit but the difference between the closed accounts have lesser mean credit card limit. From this we can understand that, if a customer with age between 48 to 58, and if their credit limit is less than 7750, there is chance they might close the account. similarly we have average credit limits for all the age groups, as the age increased beyund 58, there is higher tendency that they will close the acount. The age group 18 to 28 has lot of difference between their mean credit limit based on the status of the account. within this age group 18 and 28, the closed account customers only have a average credit limit of 3000. From the histogram it is evident that, there are equal amount of closed anc active accounts for the age groups 18 to 28 and 58 to 68. The more number of closed accounts for the customers who are in the age group 58 and 68.

What is the relationship between card type, total spend amount, and customer status?

![image](https://github.com/user-attachments/assets/441c9e12-3231-48ec-a6be-98be64508638)

![image](https://github.com/user-attachments/assets/911e56c3-2a6e-45c4-bfbe-41b1db98d496)

There are three types of cards. Blue, silver and gold. For the blue card, overall, the amount of average total spent last year is greater for the accounts that are active. The gold card has the higher average total amount spent in the last year, so for the blue card customers if the average amount spent in the last year is less than 3000, it might mean they may close the account. similarly, for the gold card customers, if they spend morethan 4000, it means he may not close the account. Similarly from the box plot it is evident that there are outliers in the dataset that might be deviating from the mean values as we seen in the column plot. For all the card types, the active account customers have a similar spending behavour in the last one year so does the customers who closed account.




## Machine Learning Modeling
Customer status was predicted using three classification algorithms.
The credit_card_df data was split into training and test sets.
A feature engineering pipeline was defined using tidymodels::recipe that includes transformations for numeric and categorical features.
The recipe was prepared with the training data and then applied to the same training data.
Model 1: Logistic Regression
The parsnip package within the tidymodels framework was used to create, fit, and extract a logistic regression model.
Visualized the importance of features in the trained logistic regression model.
The yardstick package within the tidymodels framework was used to make predictions and evaluate the performance of the trained logistic regression model. Predictions were bound to the test dataset, evaluation metrics were computed, and the ROC curve was visualized.
Model 2: Decision Tree Classification
The rpart package was used to train a decision tree classifier.
The model was tuned using grid search to find the optimal hyperparameters.
Visualized the decision tree and the importance of features.
Model 3: Random Forest Classification
The ranger package was used to train a random forest classifier.
The model was tuned using random search to find the optimal hyperparameters.
Visualized the importance of features in the trained random forest model.
## Results and Discussion
EDA: 
the analysis revealed that number of closed accounts are always lower than the number of active accounts in relation to the number of dependents on the customer. The customers with 2 or 3 three dependents are stable than those with other number of dependents as the number of dependents increased beyond 3 the number of closed accounts increased, this signifies the importance of family situtations and their financial goals. Moreover, there is a clear relationship emerges between customer status, total spending in the past year, and sales representative's number of contacts. Closed account holders tend to have lower spending in the last year compared to active account holders. Notably, customers spending less than the 4500 average are at risk of closing their accounts. The frequency of sales representative contact also influences spending behaviors, with an increase in contacts the customer tends to decease their spending. Employment status is a critical factor that influences account status. Full-time employees have  higher retention rates compared to part-time job holders and self-employed individuals. Similarly, the higher number of closed accounts are associated with part-time job holders indicating that they might close the accounts.

The age group of 38 to 48 exhibits the highest mean credit card limit and active accounts. However, the likelihood of account closure increases beyond the age of 58. Additionally, the 18 to 28 age group displays considerable variations in credit limits based on account status, with closed accounts having significanlty lower limits. Blue cardholders demonstrate higher spending tendencies, while gold cardholders exhibit higher spending thresholds before considering account closure. Notably, if blue cardholders spend less than 3000 annually, they may be inclined to close their accounts, while gold cardholders spending over 4000 are more likely to retain their accounts.
These findings are crucial for the business as they provide valuable insights into customer behavior and potential churn indicators. By understanding these patterns, the company can develop targeted retention strategies, tailor offers and services to better meet customer needs, and optimize sales representative interactions to enhance customer satisfaction and loyalty. Establishing the need for these recommendations underscores the significance of the analysis in addressing the company's pressing business problem and ensuring its future success in the competitive market.

Modeling Results:
Three models are developed and tested. The first model is logistic rgression, 2) Decision Tree Classification and 3) Random forests classifier.according to logistic regression, the top 4 important features are transactions last year, total spend last year utlilization ratio and card type silver. According to decision trees and random forests the top 4 important features are total spend last year, transactions last year, transactions ratio q4 and q1 and utlization ratio. The ROC curve is plot between sensitivity(TRP) and 1-specificity (FPR) the centraldiagonal linerepresents a perfect useless model where a curve that is incline along th left hand side on the y axis and horizontal line on the top representng 100% sensitivity and 100% specificity. The logistic regression model has lowest sensitivity and it is more closer to the diagonal line than that of the dt or rf models. As the model complexity is increased the curve is more closer to the left hand side of the plt. Overall the Random forest classifier outperformed rest of the models. As it is more closer to the left side of the plot making is more perfect model. The accuracy and AUC of DT model are 0.84 and 0.94. RF Classifier accuracy is 0.94 and AUC is 0.98.

The best performing model is random forest classifier which has shown highest accuracy and area under the curve for the test dataset. This model accurately differentiating between closed and active accounts. The higher AUC value indicated better performance with a value of 1 representing a perfect model, in case of the best model RF classifier, the AUC is 0.98 indicating excellent predicitve capacity on the unseen data. the model can correctly identify the customers who are about to close the account with an accuracy rate of 98%.
## Recommendations
Based on this analysis, the company should fous on the customer retention plans and strategies. they should consider developing random forest classification  as it  outperformed other models in terms of area under curve (AUC). The potential is also there and identifying outliers, especially in the blue card type, is important to refine the model and enhance forecast performance. Furthermore, feature engineering techniques such as creating interaction terms or finding new features based on performance status and age can better capture complex relationships in data Scientists Collaboration between domain and experts is encouraged to use domain knowledge through data analysis techniques, and will facilitate decision making and strategic efficiency. These recommendations aim to improve the accuracy of the classification model and what effectiveness has increased, ultimately improving credit card management and customer satisfaction.
