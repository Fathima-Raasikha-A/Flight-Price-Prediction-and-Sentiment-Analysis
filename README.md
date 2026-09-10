# Flight Price Prediction & Passenger Satisfaction Analysis

Two related ML tasks on airline data: a regression model predicting ticket prices, and a classification model predicting passenger satisfaction.

## Part 1: Flight Price Prediction (Regression)

### Dataset
`Flight_Price.csv` — airline, route, departure/arrival time, duration, stops, and price.

### Approach
- Cleaned missing values with backward fill
- Engineered features: split date into day/month/year, duration into hours/minutes, departure and arrival times into hours/minutes
- Split data into training and test sets, then one-hot encoded categorical features (Airline, Source, Destination, Route, Additional_Info) and label-encoded the ordinal `Total_Stops` feature — fitting encoders on the training set only to avoid data leakage
- Trained and compared 5 regression models, tracked with MLflow: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost

### Results (Test set)

| Model | MSE | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | 5,395,655 | 2,323 | 0.745 |
| Gradient Boosting | 3,819,143 | 1,954 | 0.820 |
| Decision Tree | 3,449,229 | 1,857 | 0.837 |
| Random Forest | 2,758,802 | 1,661 | 0.870 |
| **XGBoost (best)** | **2,004,553** | **1,416** | **0.905** |

RMSE gives the error in the same unit as price — XGBoost's predictions are off by about ₹1,416 on average, and it explains ~90.5% of the variance in flight prices.

## Part 2: Passenger Satisfaction Prediction (Classification)

### Dataset
`Passenger_Satisfaction.csv` — ~104K rows covering demographics, travel type, class, and 14 service-quality ratings (wifi, seat comfort, boarding, cleanliness, etc.), with a binary satisfaction label.

### Approach
- Dropped 310 rows with missing values (`Arrival Delay in Minutes`) and non-predictive ID columns
- Label-encoded categorical features (Gender, Customer Type, Type of Travel, Class, satisfaction)
- Split data into training and test sets before oversampling, then addressed class imbalance (57%/43% split) using SMOTE on the training set only — avoiding data leakage from synthetic samples into the test set
- Trained and compared 5 classifiers, tracked with MLflow: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost

### Results (Test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.859 | 0.859 | 0.859 | 0.859 |
| AdaBoost | 0.914 | 0.915 | 0.914 | 0.914 |
| Gradient Boosting | 0.940 | 0.940 | 0.940 | 0.940 |
| Decision Tree | 0.941 | 0.942 | 0.941 | 0.941 |
| **Random Forest (best)** | **0.960** | **0.960** | **0.960** | **0.960** |

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), MLflow, Matplotlib, Seaborn

## Key Takeaways
- Systematic model comparison rather than picking one algorithm upfront
- Used MLflow to track experiments, parameters, and metrics across all model runs
- Fit encoders (one-hot, label) and SMOTE oversampling on the training set only, after the train/test split, to avoid data leakage into the test set
- Reported RMSE alongside MSE for the regression task, since RMSE is in the same unit as price and is easier to interpret
