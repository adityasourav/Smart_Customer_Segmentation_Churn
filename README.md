# Smart Customer Segmentation & Churn Prediction System

<p align="center">
  <img src="images/workflow.png" width="1000">
</p>

<p align="center">
Machine Learning-based customer churn analysis and prediction system for customer retention and business insights.
</p>

---

# Project Overview

This project focuses on analyzing customer behavior and predicting customer churn using Machine Learning techniques. The system performs customer segmentation, identifies customer patterns, visualizes important insights, and predicts whether a customer is likely to leave the service.

The main objective is to help businesses improve customer retention strategies and make data-driven decisions by understanding customer behavior.

---

# Features

✅ Customer Data Analysis  
✅ Exploratory Data Analysis (EDA)  
✅ Data Cleaning and Preprocessing  
✅ Customer Segmentation  
✅ Customer Churn Prediction  
✅ Data Visualization  
✅ Model Building  
✅ Model Evaluation  
✅ Classification Report Generation  
✅ Confusion Matrix Analysis  

---

# Dataset Information

Dataset Used:

**Telco Customer Churn Dataset**

Dataset Source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Dataset Description

The dataset contains customer information from a telecommunication company and is used to predict customer churn behavior.

Dataset includes:

- Customer demographic information
- Customer account information
- Services subscribed
- Monthly charges
- Total charges
- Contract details
- Payment methods
- Customer churn information

### Target Variable

```text
Churn

Yes → Customer Leaves Service
No → Customer Stays
```

Total records:

```text
7043 Customer Records
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- Jupyter Notebook

---

# Tools Used

| Tool | Purpose |
|--------|----------|
| PyCharm | Development Environment |
| Jupyter Notebook | EDA and Model Building |
| Git | Version Control |
| GitHub | Repository Hosting |
| Streamlit | Web Application |

---

# Project Workflow

### Step 1: Data Collection

- Loaded Telco Customer Churn dataset
- Imported required libraries

### Step 2: Data Cleaning

- Handled missing values
- Checked duplicate records
- Removed inconsistencies

### Step 3: Exploratory Data Analysis (EDA)

- Customer churn analysis
- Correlation analysis
- Monthly charges analysis
- Feature relationship analysis

### Step 4: Data Preprocessing

- Feature Encoding
- Feature Scaling
- Train-Test Split

### Step 5: Model Building

- Trained Machine Learning classification model
- Learned customer patterns

### Step 6: Model Evaluation

- Generated confusion matrix
- Generated classification report
- Calculated performance metrics

### Step 7: Prediction System

- Predict whether a customer will churn or stay

---

## Workflow Diagram

<p align="center">
<img src="images/workflow.png" width="950">
</p>

---

# Project Structure

```text
Customer_Churn_prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── Customer-Churn.csv
│
├── notebook/
│   ├── EDA.ipynb
│   └── Model_Building.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── prediction.py
│   └── segmentation.py
│
├── images/
│   ├── workflow.png
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── monthly_charges.png
│   ├── scatter_plot.png
│   ├── cus_churn.png
│   ├── cus_stay.png
│   ├── confusion_matrix.png
│   └── classification_report.png
│
└── README.md
```

---

# Project Visualizations

---

## Customer Churn Distribution

<p align="center">
<img src="images/churn_distribution.png" width="700">
</p>

Shows customer churn distribution.

---

## Correlation Heatmap

<p align="center">
<img src="images/correlation_heatmap.png" width="700">
</p>

Displays relationships among multiple variables.

---

## Monthly Charges Analysis

<p align="center">
<img src="images/monthly_charges.png" width="700">
</p>

Shows impact of monthly charges on customer behavior.

---

## Scatter Plot Analysis

<p align="center">
<img src="images/scatter_plot.png" width="700">
</p>

Displays feature relationships.

---

## Customer Churn Analysis

<p align="center">
<img src="images/cus_churn.png" width="700">
</p>

Shows customer churn behavior patterns.

---

## Customer Retention Analysis

<p align="center">
<img src="images/cus_stay.png" width="700">
</p>

Shows customer retention insights.

---

## Confusion Matrix

<p align="center">
<img src="images/confusion_matrix.png" width="700">
</p>

Illustrates model prediction performance.

---

## Classification Report

<p align="center">
<img src="images/classification_report.png" width="700">
</p>

Displays precision, recall and F1 score.

---

# Results

The machine learning model successfully predicts customer churn behavior and identifies important customer patterns.

Performance metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# Future Improvements

- Hyperparameter tuning
- Improve prediction accuracy
- Deploy using Streamlit Cloud
- Real-time analytics dashboard
- Add multiple ML models

---
