# Flight Price Prediction & Passenger Satisfaction Analysis

Two related ML tasks on airline data: a regression model predicting ticket prices,
and a classification model predicting passenger satisfaction.

## Part 1: Flight Price Prediction (Regression)

### Dataset
`Flight_Price.csv` — airline, route, departure/arrival time, duration, stops, and price.

### Approach
- Cleaned missing values with backward fill
- Engineered features: split date into day/month/year, duration into hours/minutes,
  departure and arrival times into hours/minutes
- One-hot encoded categorical features (Airline, Source, Destination, Route,
  Additional_Info); label-encoded the ordinal Total_Stops feature
- Trained and compared 5 regression models, tracked with **MLflow**:
  Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost

### Results (Test set)
| Model              | R²    | MSE       |
|---------------------|-------|-----------|
| Linear Regression    | 0.745 | 5,395,655 |
| Decision Tree        | 0.832–0.843 | ~3.3–3.6M |
| Random Forest        | 0.869–0.872 | ~2.7–2.8M |
| Gradient Boosting     | 0.819 | 3,820,407 |
| **XGBoost (best)**   | **0.905** | **2,004,553** |

## Part 2: Passenger Satisfaction Prediction (Classification)

### Dataset
`Passenger_Satisfaction.csv` — ~104K rows covering demographics, travel type, class,
and 14 service-quality ratings (wifi, seat comfort, boarding, cleanliness, etc.),
with a binary satisfaction label.

### Approach
- Dropped rows with missing values and non-predictive ID columns
- Label-encoded categorical features (Gender, Customer Type, Type of Travel, Class)
- Addressed class imbalance (57%/43% split) using **SMOTE** oversampling
- Trained and compared 5 classifiers, tracked with **MLflow**:
  Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost

### Results (Test set)
| Model                  | Accuracy | Precision | Recall | F1    |
|--------------------------|----------|-----------|--------|-------|
| Logistic Regression       | 0.870    | 0.870     | 0.870  | 0.870 |
| Gradient Boosting         | 0.937    | 0.937     | 0.937  | 0.937 |
| AdaBoost                  | 0.909    | 0.909     | 0.909  | 0.909 |
| Decision Tree             | 0.941    | 0.941     | 0.941  | 0.941 |
| **Random Forest (best)** | **0.960**| **0.960** | **0.960** | **0.960** |

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), MLflow, Matplotlib, Seaborn

## Key Takeaways
- Systematic model comparison rather than picking one algorithm upfront
- Used MLflow to track experiments, parameters, and metrics across all model runs
- Applied SMOTE to address class imbalance before training classifiers
