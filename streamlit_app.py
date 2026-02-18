import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #ff7f0e;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-box {
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
}
.high-churn {
    background-color: #ffebee;
    color: #c62828;
    border: 2px solid #ffcdd2;
}
.low-churn {
    background-color: #e8f5e9;
    color: #2e7d32;
    border: 2px solid #c8e6c9;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.write("Predict customer churn using advanced machine learning algorithms")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Model Comparison", "Prediction", "EDA"])

if page == "Home":
    st.markdown("<h2 style='text-align: center; color: #1f77b4;'>Welcome to Customer Churn Analysis</h2>", unsafe_allow_html=True)
    
    # Show actual churn metrics from dataset
    try:
        df = pd.read_csv('Customer Churn.csv')
        churn_count = (df['Churn'] == 'Yes').sum()
        retained_count = (df['Churn'] == 'No').sum()
        total_customers = len(df)
        churn_rate = (churn_count / total_customers) * 100
            
        # Display metrics
        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            st.metric(label="Total Customers", value=f"{total_customers:,}")
        with col_mid:
            st.metric(label="Churn Rate", value=f"{churn_rate:.1f}%")
        with col_right:
            st.metric(label="Retained Customers", value=f"{retained_count:,}")
            
    except:
        st.write("Metrics based on customer churn analysis")
    
    st.markdown("---")  # Divider line
    
    st.markdown("<h2 style='color: #1f77b4;'>Project Overview</h2>", unsafe_allow_html=True)
    st.write("This machine learning project aims to predict customer churn for a telecommunications company using advanced algorithms. The project applies various machine learning techniques to identify patterns in customer behavior that indicate a likelihood to discontinue services.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Business Context</h2>", unsafe_allow_html=True)
    st.write("Customer churn is a critical metric for telecommunications companies, as acquiring new customers costs significantly more than retaining existing ones. This project focuses on identifying at-risk customers early to enable targeted retention strategies.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Problem Statement</h2>", unsafe_allow_html=True)
    st.write("Customer churn represents a significant challenge for telecommunications companies, with industry studies showing that acquiring a new customer costs 5-25 times more than retaining an existing one. In today's competitive market, understanding and predicting customer behavior is crucial for sustainable growth and profitability.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Dataset Information</h2>", unsafe_allow_html=True)
    st.write("This project utilizes the Telco Customer Churn dataset containing 7,043 customer records with 21 features including demographics, services subscribed, account information, and tenure. Key features analyzed include contract type, payment methods, service usage patterns, and customer support interactions.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Methodology</h2>", unsafe_allow_html=True)
    st.write("Three different machine learning algorithms were employed to build predictive models:")
    st.write("• **Logistic Regression**: Interpretable linear model")
    st.write("• **Support Vector Machine**: Effective for complex patterns")
    st.write("• **Random Forest**: Ensemble method with high accuracy")
    
    st.markdown("---")  # Divider line
    
    st.markdown("<h2 style='color: #1f77b4;'>Key Features</h2>", unsafe_allow_html=True)
    st.write("• **Real-time Prediction**: Instantly assess churn risk for individual customers")
    st.write("• **Model Comparison**: Compare performance across different algorithms")
    st.write("• **Exploratory Data Analysis**: Visualize key factors affecting customer retention")
    st.write("• **Interactive Dashboard**: User-friendly interface for business stakeholders")
    
    st.markdown("<h2 style='color: #1f77b4;'>Technical Implementation</h2>", unsafe_allow_html=True)
    st.write("• **Data Preprocessing**: Handling missing values, encoding categorical variables")
    st.write("• **Feature Engineering**: Selecting relevant attributes for prediction")
    st.write("• **Model Training**: Applying multiple algorithms for optimal results")
    st.write("• **Performance Evaluation**: Using accuracy, precision, recall, and F1-score metrics")
    
    st.markdown("<h2 style='color: #1f77b4;'>Technology Stack</h2>", unsafe_allow_html=True)
    st.write("Built using Python with scikit-learn for machine learning, pandas and numpy for data manipulation, matplotlib and seaborn for visualization, and Streamlit for the interactive dashboard. The solution demonstrates proficiency in the complete data science pipeline from preprocessing to deployment.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Solution Impact</h2>", unsafe_allow_html=True)
    st.write("Early identification of at-risk customers enables proactive intervention strategies, potentially saving significant revenue. Businesses can leverage targeted retention campaigns, personalized offers, and improved customer service to reduce churn rates and increase customer lifetime value.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Project Objective</h2>", unsafe_allow_html=True)
    st.write("Develop a predictive model to identify customers at risk of churning, enabling businesses to take proactive measures to retain valuable customers.")
    
    st.markdown("<h2 style='color: #1f77b4;'>Project Details</h2>", unsafe_allow_html=True)
    st.markdown("**📊 Dataset:** Telco Customer Churn")
    
    st.markdown("<h2 style='color: #1f77b4;'>Developer</h2>", unsafe_allow_html=True)
    st.markdown("- Mahir Khan Pathan")
        
elif page == "Model Comparison":
    st.subheader("Model Performance Comparison")
    
    # Load the models and show their performance metrics
    try:
        # These are the actual metrics from your project
        metrics_data = {
            'Algorithm': ['Logistic Regression', 'SVM', 'Random Forest'],
            'Accuracy': [0.784, 0.788, 0.809],
            'Precision': [0.756, 0.762, 0.776],
            'Recall': [0.837, 0.839, 0.868],
            'F1-Score': [0.794, 0.799, 0.819]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Updated styling for better visibility in both light and dark mode
        styled_df = df_metrics.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}'
        }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#81C784')  # Better color for dark mode
        
        st.dataframe(styled_df)
        
        # Add the performance metrics explanation after the table
        st.markdown("### About Performance Metrics")
        st.write("Below are the performance metrics for the three machine learning models used in this churn prediction project.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Accuracy**: The proportion of correct predictions among all predictions")
            st.markdown("**Precision**: The proportion of positive predictions that were actually correct")
        with col2:
            st.markdown("**Recall**: The proportion of actual positives that were correctly identified")
            st.markdown("**F1-Score**: The harmonic mean of precision and recall, providing a balanced measure")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_metrics))
        width = 0.15
        
        ax.bar(x - width*1.5, df_metrics['Accuracy'], width, label='Accuracy', alpha=0.8)
        ax.bar(x - width*0.5, df_metrics['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x + width*0.5, df_metrics['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width*1.5, df_metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df_metrics['Algorithm'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
        
        st.write("### Model Characteristics")
        characteristics = {
            "Logistic Regression": "Good interpretability, linear relationships, works well with binary classification problems, provides probability estimates",
            "SVM": "Effective for complex decision boundaries, robust against overfitting, works well with high-dimensional spaces",
            "Random Forest": "Handles non-linear patterns, provides feature importance, reduces overfitting through ensemble averaging, handles missing values well"
        }
        
        for model, desc in characteristics.items():
            st.write(f"**{model}**: {desc}")
            
    except Exception as e:
        st.error(f"Error loading model comparison: {str(e)}")

elif page == "Prediction":
    st.subheader("Customer Churn Prediction")
    
    # Load models and scaler
    try:
        lr_model = joblib.load('logistic_regression_model.pkl')
        svm_model = joblib.load('svm_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        st.write("### Enter Customer Information")
        
        # Input fields for customer data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.slider('Tenure (Months)', 0, 72, 24, step=1, help="Number of months the customer has stayed with the company")
            monthly_charges = st.number_input('Monthly Charges ($)', min_value=18.0, max_value=120.0, value=70.0, step=1.0, help="Amount charged to the customer monthly")
            
        with col2:
            total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=9000.0, value=1500.0, step=10.0, help="Total amount charged to the customer")
            contract_type = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
            
        with col3:
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
            paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        
        # Test example
        st.info("💡 **Test Example**: For a customer with 24 months tenure, $70 monthly charges, $1500 total charges, Month-to-month contract, DSL internet, No tech support, and Paperless billing, enter these values above and click 'Predict Churn Risk'")
        
        # Show the logical relationship between inputs
        estimated_total_calculation = tenure * monthly_charges
        st.caption(f"*Note: Typically, Total Charges ≈ Tenure × Monthly Charges ({tenure} × ${monthly_charges:.2f} = ${estimated_total_calculation:.2f}) + additional fees*")
        
        # Validate input values
        if monthly_charges > total_charges and tenure > 1:
            st.warning(f"⚠️ **Warning**: Total charges (${total_charges:.2f}) is less than monthly charges (${monthly_charges:.2f}) for a customer with {tenure} months tenure. This seems unrealistic.")
        elif total_charges < monthly_charges and tenure == 0:
            st.warning(f"⚠️ **Warning**: Total charges (${total_charges:.2f}) is less than monthly charges (${monthly_charges:.2f}). This seems unusual even for new customers.")
        
        # Prepare input data
        if st.button('Predict Churn Risk'):
            # Encode the input data the same way as training data
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract_type],
                'InternetService': [internet_service],
                'TechSupport': [tech_support],
                'PaperlessBilling': [paperless_billing]
            })
            
            # Apply same encoding as in training
            input_encoded = input_data.copy()
            input_encoded['Contract'] = input_encoded['Contract'].map({
                'Month-to-month': 0,
                'One year': 1,
                'Two year': 2
            })
            
            input_encoded['InternetService'] = input_encoded['InternetService'].map({
                'DSL': 0,
                'Fiber optic': 1,
                'No': 2
            })
            
            input_encoded['TechSupport'] = input_encoded['TechSupport'].map({
                'No': 0,
                'Yes': 1,
                'No internet service': 2
            })
            
            input_encoded['PaperlessBilling'] = input_encoded['PaperlessBilling'].map({
                'No': 0,
                'Yes': 1
            })
            
            # Scale the input data
            input_scaled = scaler.transform(input_encoded)
            
            # Make predictions
            lr_prob = lr_model.predict_proba(input_scaled)[0][1]  # Probability of churn
            svm_prob = svm_model.predict_proba(input_scaled)[0][1]  # Probability of churn
            rf_prob = rf_model.predict_proba(input_scaled)[0][1]  # Probability of churn
            
            # Predict classes
            lr_pred = lr_model.predict(input_scaled)[0]
            svm_pred = svm_model.predict(input_scaled)[0]
            rf_pred = rf_model.predict(input_scaled)[0]
            
            # Convert to labels
            lr_result = "Churn" if lr_pred == 1 else "No Churn"
            svm_result = "Churn" if svm_pred == 1 else "No Churn"
            rf_result = "Churn" if rf_pred == 1 else "No Churn"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Logistic Regression**")
                prob_text = f"**Churn Probability: {lr_prob:.2%}**"
                if lr_prob > 0.5:
                    st.markdown(f'<div class="prediction-box high-churn">{prob_text}<br>Prediction: {lr_result}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box low-churn">{prob_text}<br>Prediction: {lr_result}</div>', unsafe_allow_html=True)
                    
            with col2:
                st.write("**Support Vector Machine**")
                prob_text = f"**Churn Probability: {svm_prob:.2%}**"
                if svm_prob > 0.5:
                    st.markdown(f'<div class="prediction-box high-churn">{prob_text}<br>Prediction: {svm_result}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box low-churn">{prob_text}<br>Prediction: {svm_result}</div>', unsafe_allow_html=True)
                    
            with col3:
                st.write("**Random Forest**")
                prob_text = f"**Churn Probability: {rf_prob:.2%}**"
                if rf_prob > 0.5:
                    st.markdown(f'<div class="prediction-box high-churn">{prob_text}<br>Prediction: {rf_result}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box low-churn">{prob_text}<br>Prediction: {rf_result}</div>', unsafe_allow_html=True)
            
            # Overall prediction based on majority vote
            predictions = [lr_pred, svm_pred, rf_pred]
            majority_pred = 1 if sum(predictions) >= 2 else 0
            majority_result = "Churn" if majority_pred == 1 else "No Churn"
            
            st.subheader("Overall Prediction")
            avg_prob = (lr_prob + svm_prob + rf_prob) / 3
            overall_text = f"**Average Churn Probability: {avg_prob:.2%}**"
            if avg_prob > 0.5:
                st.markdown(f'<div class="prediction-box high-churn" style="font-size: 1.5rem;">{overall_text}<br><span style="font-size: 1.8rem;">OVERALL: {majority_result}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box low-churn" style="font-size: 1.5rem;">{overall_text}<br><span style="font-size: 1.8rem;">OVERALL: {majority_result}</span></div>', unsafe_allow_html=True)
                
    except FileNotFoundError:
        st.error("Models not found. Please run the save_models.py script first.")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    
    # Load the dataset
    df = pd.read_csv('Customer Churn.csv')
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = df["TotalCharges"].astype("float")
    
    st.write("### Dataset Overview")
    st.dataframe(df.head())
    
    st.write("### Dataset Statistics")
    st.dataframe(df.describe())
    
    # Churn Distribution
    st.write("### Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Customer Churn Distribution')
    st.pyplot(fig)
    
    # Interpretation
    st.write("**Insight**: The pie chart shows the proportion of customers who churned versus those who remained. This helps understand the class distribution in our dataset.")
    
    # Feature distributions by churn
    st.write("### Feature Distributions by Churn Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tenure Distribution**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Churn', y='tenure', ax=ax)
        ax.set_title('Tenure by Churn Status')
        st.pyplot(fig)
        st.write("**Insight**: Customers who churn tend to have shorter tenure compared to loyal customers.")
    
    with col2:
        st.write("**Monthly Charges Distribution**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=ax)
        ax.set_title('Monthly Charges by Churn Status')
        st.pyplot(fig)
        st.write("**Insight**: Churned customers often have higher monthly charges compared to loyal customers.")
    
    # Contract vs Churn
    st.write("### Contract Type vs Churn")
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contract_churn.plot(kind='bar', ax=ax)
    ax.set_title('Churn Rate by Contract Type (%)')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Churn')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Interpretation
    st.write("**Insight**: Customers with month-to-month contracts have a significantly higher churn rate compared to those with one-year or two-year contracts.")
    
    # Correlation heatmap
    st.write("### Feature Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in df.columns:
        # Map Churn to 0/1 for correlation
        df_corr = df.copy()
        df_corr['Churn'] = df_corr['Churn'].map({'No': 0, 'Yes': 1})
        corr_matrix = df_corr[numeric_cols + ['Churn']].corr()
    else:
        corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    
    # Interpretation
    st.write("**Insight**: The heatmap shows correlations between numerical features. Positive correlations appear in red, negative in blue. Strong correlations with Churn indicate important predictive features.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.write("Customer Churn Prediction Project")
st.sidebar.write("Built with Streamlit & Scikit-learn")