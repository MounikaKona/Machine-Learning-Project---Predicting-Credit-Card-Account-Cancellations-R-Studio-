# Load data
library(tidyverse)
library(ggplot2)
library(dplyr)

credit_card_df <- readRDS(url('https://gmubusinessanalytics.netlify.app/data/credit_card_df.rds'))
# View data
credit_card_df

#####Exploratory Data Analysis
#####Question1: What is the relationship between the customer status and the number of dependents in the family
# The below code adjusts the figure output size in the notebook

options(repr.plot.width=11, repr.plot.height=8)
names(credit_card_df)


summary_stats <- credit_card_df %>%
  group_by(dependents, customer_status) %>%
  summarise(count = n()) %>%
  ungroup()  

print(summary_stats)
ggplot(summary_stats, aes(x = factor(dependents), y = count, fill = customer_status)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Distribution of Customer Status by Number of Dependents",
       x = "Number of Dependents",
       y = "Count") +
  scale_fill_manual(values = c("skyblue", "blue", "green", "red")) +
  theme_minimal()

###Question2: Is there any relationship between the customer status and the customers who were contacted by the company last year and those who are not?

cleaned_data <- credit_card_df %>%
  
  filter(!is.na(contacted_last_year) & !is.na(customer_status))

spending_summary <- cleaned_data %>%
  group_by(contacted_last_year, customer_status) %>%
  summarise(
    mean_total_spend = mean(total_spend_last_year),
    median_total_spend = median(total_spend_last_year),
    sd_total_spend = sd(total_spend_last_year)
  )

print(spending_summary)

spending_summary_long <- spending_summary %>%
  pivot_longer(cols = c(mean_total_spend, median_total_spend),
               names_to = "Statistic", values_to = "Value")

ggplot(spending_summary_long, aes(x = contacted_last_year, y = Value, color = Statistic, group = Statistic)) +
  geom_line() +
  geom_point() +
  facet_wrap(~customer_status) +
  labs(x = "Contacted Last Year", y = "Total Spend", title = "Total Spend Based on Contact Status and Customer Status",
       color = "Statistic") +
  theme_minimal()

####Question 3: What is the relationship between employment status and customer status?

cleaned_data <- credit_card_df %>%
  
  filter(!is.na(employment_status) & !is.na(customer_status))


customer_status_summary <- cleaned_data %>%
  group_by(employment_status, customer_status) %>%
  summarize(
    count = n()
  )


print(customer_status_summary)

customer_status_summary_long <- customer_status_summary %>%
  pivot_longer(cols = c(count),
               names_to = "Statistic", values_to = "Value")


ggplot(customer_status_summary_long, aes(x = employment_status, y = Value, fill = customer_status)) +
  geom_bar(stat = "identity") +
  labs(x = "Employment Status", y = "Count", title = "Customer Status by Employment Status",
       fill = "Customer Status") +
  theme_minimal()

###Question4: Is there a relationship between age, their credit card limit and customer status?

cleaned_data <- credit_card_df %>%
  filter(!is.na(age) & !is.na(customer_status))

cleaned_data <- cleaned_data %>%
  mutate(age_group = cut(age, breaks = seq(18, 100, by = 10)))


summary_stats <- cleaned_data %>%
  group_by(age_group, customer_status) %>%
  summarize(
    mean_credit_limit = mean(credit_limit),
    sem_credit_limit = sd(credit_limit) / sqrt(n())
  )


print(summary_stats)

ggplot(summary_stats, aes(x = age_group, y = mean_credit_limit, fill = customer_status)) +
  geom_col(position = "dodge", width = 0.5) +
  geom_errorbar(aes(ymin = mean_credit_limit - sem_credit_limit, 
                    ymax = mean_credit_limit + sem_credit_limit),
                position = position_dodge(width = 0.5), width = 0.2, color = "black") +
  labs(
    x = "Age Group",
    y = "Mean Credit Limit",
    title = "Mean Credit Limit by Age Group with Standard Error of the Mean (SEM) Error Bars, Grouped by Customer Status"
  ) +
  theme_minimal()



cleaned_data <- credit_card_df %>%
  filter(!is.na(age) & !is.na(credit_limit) & !is.na(customer_status))

ggplot(cleaned_data, aes(x = credit_limit, fill = customer_status)) +
  geom_histogram(binwidth = 500, color = "black") +
  facet_wrap(~cut(age, breaks = seq(18, 100, by = 10)), scales = "free_x", nrow = 3) +
  labs(
    x = "Credit Limit",
    y = "Frequency",
    title = "Distribution of Credit Limits by Age Group and Customer Status"
  ) +
  scale_fill_manual(values = c("skyblue", "red", "green", "blue")) +  
  theme_minimal()

###Question5 : what is the relationship between card type, total spend amount and customer status?

cleaned_data <- credit_card_df %>%
  filter(!is.na(card_type) & !is.na(customer_status) & !is.na(total_spend_last_year))


summary_stats <- cleaned_data %>%
  group_by(card_type, customer_status) %>%
  summarize(
    mean_total_spend = mean(total_spend_last_year),
    median_total_spend = median(total_spend_last_year),
    min_total_spend = min(total_spend_last_year),
    max_total_spend = max(total_spend_last_year),
    n = n()
  )

print(summary_stats)


ggplot(summary_stats, aes(x = card_type, y = mean_total_spend, fill = customer_status)) +
  geom_col(position = "dodge", width = 0.5) +
  labs(
    x = "Card Type",
    y = "Mean Total Spend",
    title = "Mean Total Spend by Card Type and Customer Status"
  ) +
  theme_minimal()
ggplot(cleaned_data, aes(x = card_type, y = total_spend_last_year, fill = customer_status)) +
  geom_boxplot() +
  labs(
    x = "Card Type",
    y = "Total Spend Last Year",
    title = "Distribution of Total Spend Across Card Types by Customer Status"
  ) +
  theme_minimal()

####ML Models####
###Install ML Packages

library(vip)
library(rsample)
library(tidymodels)
library(recipes)
library(parsnip)
library(vip)
names(credit_card_df)

##Data Resampling
#LOGISTICREGRESSION DTS AND RANDOM FORESTS
# First model is Logistic Regression


set.seed(314)

credictcard_split <- initial_split(credit_card_df, prop = 0.75, 
                                   strata = customer_status)
# Generate a training data frame
creditcard_training <- 
  credictcard_split %>% 
  training()

# View results


creditcard_test <- 
  credictcard_split %>% 
  testing()


num_folds = 5

folds = vfold_cv(creditcard_training, v= num_folds)

####Feature Engineering Pipeline

names(credit_card_df)
str(credit_card_df)
credit_card_recipe <- recipe(customer_status ~ ., data = credit_card_df) %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes())


credit_card_recipe %>% 
  prep(training = creditcard_training) %>% 
  bake(new_data = NULL)

###MOdel 1Training
#LOGISTIC REGRESSION

logistic_model <- 
  logistic_reg() %>% 
  set_engine('glm') %>% 
  set_mode('classification')

credit_card_workflow <- workflow() %>%
  add_model(logistic_model) %>%
  add_recipe(credit_card_recipe)

model_logi_fit <- fit(credit_card_workflow, data = creditcard_training)
logi_trained_model <- 
  model_logi_fit %>% 
  extract_fit_parsnip()
options(repr.plot.width=11, repr.plot.height=8)

vip(logi_trained_model)

####Model 1 performance evaluation

library(yardstick)

predictions_categories <- 
  predict(model_logi_fit, new_data = creditcard_test)

predictions_categories

predictions_probabilities <- 
  predict(model_logi_fit, new_data = creditcard_test, type = 'prob')

predictions_probabilities
test_results <- 
  creditcard_test %>% 
  select(customer_status) %>% 
  bind_cols(predictions_categories) %>% 
  bind_cols(predictions_probabilities)
test_results
spec(test_results,
     truth = customer_status, 
     estimate = .pred_class)

roc_curve(test_results, 
          truth = customer_status,
          .pred_closed_account)

roc_curve(test_results, 
          truth = customer_status, 
          .pred_closed_account) %>% 
  autoplot()

####Model 2 Training
#  Decision Tree CLASSIFICATION

d_tree_model <- decision_tree(cost_complexity = tune(),
                              tree_depth = tune(),
                              min_n = tune()) %>% 
  set_engine('rpart') %>% 
  set_mode('classification')

tree_workflow <- workflow() %>% 
  add_model(d_tree_model) %>% 
  add_recipe(credit_card_recipe)

tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          min_n(), 
                          levels =2)
tree_grid <- grid_regular(parameters(d_tree_model), 
                          levels = 2)

set.seed(314)

tree_tuning <- tree_workflow %>% 
  tune_grid(resamples = folds,
            grid = tree_grid)

tree_tuning %>% show_best(metric = 'roc_auc')

best_tree <- tree_tuning %>% 
  select_best(metric = 'roc_auc')

# View the best tree parameters

best_tree
final_tree_workflow <- tree_workflow %>% 
  finalize_workflow(best_tree)

tree_wf_fit <- final_tree_workflow %>% 
  fit(data = creditcard_training)
tree_fit <- tree_wf_fit %>% 
  extract_fit_parsnip()

vip(tree_fit)

library(rpart.plot)
rpart.plot(tree_fit$fit, roundint = FALSE, extra = 2)
credictcard_split

##Model 2 Performance evaluation

tree_last_fit <- final_tree_workflow %>% 
  last_fit(credictcard_split)
tree_last_fit %>% collect_predictions() %>% 
  roc_curve(truth  = customer_status, .pred_closed_account) %>% 
  autoplot()
tree_last_fit %>% collect_metrics()

###Model 3 Training

# RANDOM FORESTS

rf_model <- rand_forest(mtry = tune(),
                        trees = tune(),
                        min_n = tune()) %>% 
  set_engine('ranger', importance = "impurity") %>% 
  set_mode('classification')

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(credit_card_recipe)

set.seed(314)

rf_grid <- grid_random(mtry() %>% range_set(c(2, 4)),
                       trees(),
                       min_n(),
                       size = 10)


set.seed(314)
library(ranger)
library(magrittr)
rf_tuning <- rf_workflow %>% 
  tune_grid(resamples = folds,
            grid = rf_grid)
rf_tuning %>% show_best(metric = 'roc_auc')
best_rf <- rf_tuning %>% 
  select_best(metric = 'roc_auc')

# View the best parameters

best_rf
final_rf_workflow <- rf_workflow %>% 
  finalize_workflow(best_rf)
rf_wf_fit <- final_rf_workflow %>% 
  fit(data = creditcard_training)
rf_fit <- rf_wf_fit %>% 
  extract_fit_parsnip()
library(vip)
vip(rf_fit)


rf_last_fit <- final_rf_workflow %>% 
  last_fit(credictcard_split)
rf_last_fit %>% collect_metrics()

##Model 3 performance evaluation

rf_last_fit %>% collect_predictions() %>% 
  roc_curve(truth  = customer_status, .pred_closed_account) %>% 
  autoplot()

rf_predictions <- rf_last_fit %>% collect_predictions()

conf_mat(rf_predictions, truth = customer_status, estimate = .pred_class)

