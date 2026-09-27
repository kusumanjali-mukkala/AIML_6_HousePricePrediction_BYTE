# AIML_6_HousePricePrediction_BYTE

## Task 6 - House Price Prediction using Linear Regression

### 📌 Project Overview

This project implements a Machine Learning model to predict house values using **Linear Regression**.

The project was completed as part of the **B.Y.T.E by Arithmatrix AVIP 2026 Virtual Internship Program – AI/ML Task 6**.

The complete workflow includes:

- Loading the dataset
- Exploring the dataset
- Separating features and target
- Splitting the data into training and testing sets
- Training a Linear Regression model
- Evaluating the model
- Saving the trained model
- Generating predictions
- Visualizing actual vs predicted values
- Analyzing prediction residuals

---

## 🎯 Objective

The main objective of this project is to build a Linear Regression model that can predict house values based on different housing-related features.

---

## 📂 Dataset

The project uses the California Housing dataset available through Scikit-learn.

The dataset contains:

- **20,640 records**
- **8 input features**
- **1 target variable**

### Features

| Feature | Description |
|---|---|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population |
| AveOccup | Average house occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

### Target

**MedHouseVal** – Median house value.

The target values are represented in units of **$100,000**.

For example:

`3.5 ≈ $350,000`

---

## 🤖 Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value.

In this project:

**Input:** Housing features

**Output:** Predicted house value

The model learns the relationship between the input features and the target value from the training data.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Load Dataset
   ↓
Explore Dataset
   ↓
Separate Features and Target
   ↓
Train-Test Split
   ↓
Train Linear Regression Model
   ↓
Generate Predictions
   ↓
Evaluate Model
   ↓
Save Model
   ↓
Generate Visualizations