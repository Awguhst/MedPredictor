# Medical Predictor

This is a Streamlit-based machine learning application that offers multiple functionalities for health-related predictions. The app includes different pages, each targeting a specific aspect of health diagnostics and data querying:

## Symptom-Based Disease Prediction

This page allows users to select various symptoms from a multi-select dropdown, after which the app predicts potential diseases based on the selected symptoms. Additionally, users can explore the SHAP (SHapley Additive exPlanations) values, which help to understand how each symptom contributes to the model's prediction.

## Cardiovascular Disease Prediction

The second page predicts the likelihood of cardiovascular disease based on personal data, such as:
- Cholesterol level
- Blood pressure
- Age
- Weight
- And other relevant factors

Users can input their personal statistics to get a prediction on whether they are at risk for cardiovascular issues. SHAP values are also used here to help explain the contributions of each factor to the overall prediction, offering more transparency into how the model makes its assessment.

## Pneumonia and Tuberculosis Detection from X-Ray Images

This pages allows users to upload an X-ray images, and the application will predict whether the person has pneumonia or tuberculosis based on the image. The model uses deep learning techniques for image classification, trained on medical X-ray datasets specifically for pneumonia and tuberculosis detection.

## Data Exploration and Filtering

The final page allows users to query the dataset that was used to train the machine learning models. It provides options to filter the data based on different attributes, giving users the ability to explore and analyze the dataset. This feature is particularly useful for understanding how the model was trained and for gaining insights from the raw data.

---

## 🎥 **Demo**

![Streamlit app GIF](media/demo.gif)  

To try out the Medical Predictor app, visit the following link: [Open in Streamlit](https://medpredictor-mhv7x4ryaycn2rabdtkktw.streamlit.app/)

---

## Models and Datasets

This section will provide detailed information on the machine learning models used in the application, including the datasets, metrics, and other relevant details.

### Symptom-Based Disease Prediction Model

- **Dataset Used:** [Disease Prediction Dataset](https://www.kaggle.com/datasets/noeyislearning/disease-prediction-based-on-symptoms/data)  
- **Model Type:** Logistic Regression  
- **Metrics:** Accuracy: 1.0, F1-Score: 1.0
- **Description:** This model predicts diseases based on symptoms entered by the user. It was trained on a dataset containing symptom-disease relationships and uses a multi-class classification approach. SHAP values are used to explain the model’s decisions.

### Cardiovascular Disease Prediction Model

- **Dataset Used:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)
- **Model Type:** XGBoost
- **Metrics:** Accuracy: 0.74, F1-Score: 0.74
- **Hyperparameter Optimization:** Optuna
- **Description:** This model predicts the likelihood of cardiovascular disease based on personal statistics like cholesterol, blood pressure, and other relevant health data. It uses a binary classification approach to classify patients into risk groups.

### Pneumonia Detection

- **Dataset Used:** [Pneumonia Detection Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- **Model Type:** Convolutional Neural Network (CNN)
- **Metrics:** F1-Score : 0.94
- **Description:** This model is a deep learning-based CNN that classifies X-ray images to detect pneumonia. It was trained on a large dataset of X-ray images with labels indicating whether the person has pneumonia or not.

### Tuberculosis Detection 

- **Dataset Used:** [Tuberculosis Detection Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **Model Type:** Convolutional Neural Network (CNN)
- **Metrics:** F1-Score : 0.98
- **Description:** Similarly this model classifies X-ray images in order to detect tuberculosis.
---
