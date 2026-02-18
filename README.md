# 📊 Customer Churn Prediction: High-Impact ML for Telecom Retention

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## � Live Demo
Check out the interactive dashboard here: 
👉 **[customer-churn-prediction-using-python.streamlit.app](https://customer-churn-prediction-using-python.streamlit.app/)**

## �🚀 Overview
Customer churn is a critical challenge in the telecommunications industry, where acquisition costs far exceed retention costs. This project leverages **Machine Learning** to identify at-risk customers, enabling businesses to proactively deploy retention strategies. 

I developed a high-recall predictive system using a suite of algorithms (**Logistic Regression, SVM, and Random Forest**) to ensure high accuracy in identifying potential churners.

## 🛠️ Key Features
- **Real-time Prediction**: Interactive dashboard for instant churn risk assessment.
- **Explainable AI**: Visualizes factors driving churn (e.g., contract type, monthly charges).
- **Comprehensive EDA**: In-depth analysis of customer behavior patterns.
- **Model Intelligence**: Benchmarking multiple algorithms to select the most robust solution.

## 📈 Model Performance
| Algorithm | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **80.9%** | **86.8%** | **81.9%** |
| SVM | 78.8% | 83.9% | 79.9% |
| Logistic Regression | 78.4% | 83.7% | 79.4% |

## 📂 Project Structure
- `streamlit_app.py`: Main interactive dashboard application.
- `save_models.py`: Production-ready script for model training and serialization.
- `TCA_mahiri.ipynb`: Full analytical playground with deep-dive EDA and feature engineering.
- `Customer Churn.csv`: Standardized Telco dataset.

## ⚙️ How to Run Locally

### Setup & Launch
1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Train Models**: `python save_models.py`
4. **Run Dashboard**: `streamlit run streamlit_app.py`

## 👨‍💻 Developer
- **Mahir Khan Pathan** - *Machine Learning Explorer*

---
*Building data-driven solutions for real-world business impact.*