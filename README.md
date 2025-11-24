**# MedPredictor**# Machine Learning Streamlit Application

This is a Streamlit-based machine learning application that offers multiple functionalities for health-related predictions. The app includes different pages, each targeting a specific aspect of health diagnostics and data querying:

## 1. Symptom-Based Disease Prediction

This page allows users to select various symptoms from a multi-select dropdown, after which the app predicts potential diseases based on the selected symptoms. Additionally, users can explore the SHAP (SHapley Additive exPlanations) values, which help to understand how each symptom contributes to the model's prediction.

## 2. Cardiovascular Disease Prediction

The second page predicts the likelihood of cardiovascular disease based on personal data, such as:
- Cholesterol level
- Blood pressure
- Age
- Weight
- And other relevant factors

Users can input their personal statistics to get a prediction on whether they are at risk for cardiovascular issues. SHAP values are also used here to help explain the contributions of each factor to the overall prediction, offering more transparency into how the model makes its assessment.

## 3. Pneumonia Detection from X-Ray Images

This page allows users to upload an X-ray image, and the application will predict whether the person has pneumonia based on the image. The model uses deep learning techniques for image classification, trained on medical X-ray datasets.

## 4. Data Exploration and Filtering

The final page allows users to query the dataset that was used to train the machine learning models. It provides options to filter the data based on different attributes, giving users the ability to explore and analyze the dataset. This feature is particularly useful for understanding how the model was trained and for gaining insights from the raw data.

---

## Models and Datasets

This section will provide detailed information on the machine learning models used in the application, including the datasets, metrics, and other relevant details.

### 1. Symptom-Based Disease Prediction Model

- **Dataset Used:** [Insert dataset link]
- **Model Type:** Random Forest Classifier (or other models used)
- **Metrics:** Accuracy, F1-Score, AUC (Area Under Curve)
- **Hyperparameter Optimization:** Optuna (or other optimization techniques)
- **Description:** This model predicts diseases based on symptoms entered by the user. It was trained on a dataset containing symptom-disease relationships and uses a multi-class classification approach. SHAP values are used to explain the model’s decisions.

### 2. Cardiovascular Disease Prediction Model

- **Dataset Used:** [Insert dataset link]
- **Model Type:** Logistic Regression / Random Forest (or other models used)
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Hyperparameter Optimization:** Optuna (or other optimization techniques)
- **Description:** This model predicts the likelihood of cardiovascular disease based on personal statistics like cholesterol, blood pressure, and other relevant health data. It uses a binary classification approach to classify patients into risk groups.

### 3. Pneumonia Detection from X-Ray Model

- **Dataset Used:** [Insert dataset link] (e.g., Chest X-ray Dataset)
- **Model Type:** Convolutional Neural Network (CNN)
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC
- **Hyperparameter Optimization:** Optuna (or other optimization techniques)
- **Description:** This model is a deep learning-based CNN that classifies X-ray images to detect pneumonia. It was trained on a large dataset of X-ray images with labels indicating whether the person has pneumonia or not.

### 4. Data Querying and Exploration

- **Dataset Used:** [Insert dataset link]
- **Model Type:** N/A (Data Exploration Tool)
- **Description:** This page does not involve a machine learning model but allows users to interact with the underlying data. Users can filter and query the data based on various features, helping them understand the dataset better and explore relationships between features.

---
