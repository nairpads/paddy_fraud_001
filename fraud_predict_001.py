import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

st.set_page_config(page_title="Fraud Prediction App", layout="wide")
st.title("🔍 Fraud Prediction using Random Forest")

# Upload dataset
uploaded_file = st.file_uploader("Upload synthetic_fraud_dataset.csv", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Dataset Sample")
    st.dataframe(data.head())

    if 'is_fraud' not in data.columns:
        st.error("Dataset must include 'is_fraud' as the target column.")
    else:
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'location', 'device_type']
        label_encoders = {}

        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

        # Prepare features and target
        X = data.drop('is_fraud', axis=1)
        y = data['is_fraud']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'], ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

        # Save predictions to DataFrame
        results_df = X_test.copy()
        results_df['Actual'] = y_test
        results_df['Predicted'] = y_pred

        # Download predictions
        st.subheader("📥 Download Predictions")
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(results_df)
        st.download_button("Download Results as CSV", data=csv,
                           file_name='fraud_predictions.csv', mime='text/csv')

        # Optional: Save Model & Encoders
        with open('fraud_detection_model.pkl', 'wb') as f:
            joblib.dump(model, f)
        with open('label_encoders.pkl', 'wb') as f:
            joblib.dump(label_encoders, f)
        st.success("Model and Label Encoders saved locally.")

else:
    st.info("👆 Please upload a CSV file to begin.")
