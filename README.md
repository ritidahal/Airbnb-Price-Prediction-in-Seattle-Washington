
# Airbnb Price Prediction – Seattle, Washington

## 📌 Project Overview

This project focuses on predicting Airbnb listing prices in **Seattle, Washington** using data-driven modeling techniques. The goal is to identify the key factors that influence nightly prices and compare multiple predictive models to determine the best-performing approach.

The analysis applies **statistical modeling and machine learning techniques** to provide insights that can support **hosts, analysts, and platforms** in making informed pricing decisions.

---

## 🎯 Objectives

* Analyze factors influencing Airbnb listing prices in Seattle
* Perform data cleaning, transformation, and feature engineering
* Build and compare predictive models
* Evaluate model performance and interpret results
* Recommend the best model for pricing prediction

---

## 📊 Dataset

* **Source:** Inside Airbnb (Open Data)
* **Location:** Seattle, Washington
* **Observations:** ~6,800 listings
* **Features:** 79 variables including:

  * Property characteristics (beds, bedrooms, bathrooms, accommodates)
  * Availability metrics
  * Host attributes
  * Neighborhood and room type information

---

## 🧹 Data Preparation

* Removed extreme outliers using 1st and 99th percentiles
* Applied **log transformation** to price to reduce skewness
* Handled missing values using:

  * Mean imputation
  * Group-based imputation
* Standardized numerical variables
* Reduced high-cardinality categorical variables into meaningful groups
* Addressed multicollinearity using correlation analysis and VIF

---

## 🛠️ Models Used

The following models were developed and evaluated:

### 1. LASSO Regression

* Feature selection through regularization
* Strong interpretability
* Captures linear relationships effectively

### 2. Decision Tree

* Captures non-linear relationships
* Easy to interpret
* Pruned using cost-complexity analysis to avoid overfitting

### 3. Random Forest ⭐

* Ensemble learning method
* Captures complex interactions
* Best predictive performance among all models

---

## 📈 Model Performance (RMSE – Test Data)

| Model             | RMSE      |
| ----------------- | --------- |
| LASSO Regression  | ~0.37     |
| Decision Tree     | ~0.35     |
| **Random Forest** | **~0.33** |

➡️ **Random Forest** performed the best with the lowest RMSE and strong generalization.

---

## 🔍 Key Insights

* **Accommodates** is the strongest predictor of price
* **Property type** (entire home vs. room) significantly impacts pricing
* **Neighborhood location** (downtown & tourist hubs) commands higher prices
* Larger properties with more **beds and bedrooms** are priced higher
* Host experience (number of listings) also influences pricing

---

## ✅ Recommendation

The **Random Forest model** is recommended for Airbnb price prediction due to:

* Highest accuracy
* Ability to model non-linear relationships
* Strong generalization on unseen data

---

## 🚀 Future Improvements

* Incorporate seasonality and demand trends
* Include review-based features and sentiment analysis
* Add proximity to tourist attractions
* Extend the model to other cities for scalability

---

## 🧰 Tools & Technologies

* **SAS** (Data cleaning, modeling, diagnostics)
* **Regression & Machine Learning Techniques**
* **Statistical Evaluation Metrics (RMSE, R²)**


