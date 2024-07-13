import joblib
import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
import sklearn

# Name of the project
st.title("House Prices prediction application")

# Description of the project
st.write("This applicaion will predict the final price of each house.")
st.write("To predict  load your data with test values.")

#Step1: Download CSV file with test-values
uploaded_file = st.file_uploader("Load your data CSV-file", type='csv')
if uploaded_file is not None:
    test = pd.read_csv(uploaded_file)
    st.write(test.head(10))
else:
    st.stop()

#Step2: Appllying ML -model to the test-value
ml_pipeline= joblib.load("clf.pkl")

# Define the drop features
drop_features = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MasVnrType', 'Street']

#Step3: Generation of predictions
# Preprocess the test data
X_test = test.drop(columns=drop_features)

# Apply the same preprocessing steps
X_test_preprocessed = ml_pipeline.named_steps['preprocessor'].transform(X_test) 
# Make predictions on the test set
preds_test = ml_pipeline.named_steps['model'].predict(X_test_preprocessed)

# Since we applied log1p transformation on y, we need to reverse it
preds_test = np.expm1(preds_test)
#Step4: PresenVisualiztion of results
# Step 4: Visualization of results
# Display predictions
test['SalePrice'] = preds_test
st.write("Predictions:")
st.write(test[['Id', 'SalePrice']].head(10))

# Download link for the predictions
st.download_button(
    label="Download Predictions as CSV",
    data=test[['Id', 'SalePrice']].to_csv(index=False),
    file_name='predictions.csv',
    mime='text/csv',
)
