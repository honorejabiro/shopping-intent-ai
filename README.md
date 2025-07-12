# Online Shopping Purchase Prediction 

This project uses **K-Nearest Neighbors (KNN)** to predict whether an online shopping session will result in a purchase. The model is trained using a dataset of user behavior and session attributes collected from a website.

---

## Project Overview

- **Goal**: Predict whether a user will complete a purchase (Revenue = True/False)
- **Algorithm**: K-Nearest Neighbors (KNN) with `k=1`
- **Data Source**: CSV file with session attributes like page durations, bounce rates, month, and visitor type
- **Evaluation Metrics**:  
  - **Sensitivity (True Positive Rate)**  
  - **Specificity (True Negative Rate)**

---

## Features

- Preprocesses categorical data (month, visitor type, weekend)
- Splits data into training and test sets (60% train / 40% test)
- Uses Scikit-Learn’s `KNeighborsClassifier` for training
- Calculates prediction accuracy using custom sensitivity/specificity logic

---

## Example Usage

```bash
python3 shopping.py shopping.csv
```
Or with older python version

```bash
python shopping.py shopping.csv
```
