# EMI Prediction System - Deployment Guide

## 🚀 Quick Deployment

### Option 1: Local Testing
```bash
# 1. Install Python 3.8+ if not already installed
# 2. Navigate to project directory
cd emi-prediction-system

# 3. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
streamlit run emi_prediction_app.py

# 6. Open http://localhost:8501 in your browser
```

### Option 2: Streamlit Cloud Deployment

1.  **Prepare your GitHub Repository:**
    *   Ensure your project is in a public GitHub repository.
    *   The repository should contain at least: `emi_prediction_app.py`, `requirements.txt`, `mlflow_metadata.pkl`, `emi_processed_data.pkl`, `best_classification_model.pkl`, `best_regression_model.pkl`.

2.  **Go to Streamlit Cloud:**
    *   Visit [https://share.streamlit.io/](https://share.streamlit.io/) and log in with your GitHub account.

3.  **Deploy a new app:**
    *   Click on "New app" or "Deploy an app".
    *   Select your GitHub repository.
    *   For the "Main file path", enter `emi_prediction_app.py`.
    *   Click "Deploy!"

4.  **Monitor Deployment:**
    *   Streamlit Cloud will build and deploy your application.
    *   You can monitor the logs directly in the browser.

5.  **Access Your App:**
    *   Once deployed, Streamlit will provide you with a public URL for your application.

## 📝 Important Files for Deployment

*   `emi_prediction_app.py`: The main Streamlit application script.
*   `requirements.txt`: Lists all Python dependencies needed.
*   `mlflow_metadata.pkl`: Contains metadata about the MLflow experiment and models.
*   `emi_processed_data.pkl`: Contains preprocessing objects (scalers, encoders) essential for real-time inference.
*   `best_classification_model.pkl`: The saved best performing classification model.
*   `best_regression_model.pkl`: The saved best performing regression model.

## ✅ Post-Deployment Checks

*   Verify that all pages of the application load correctly.
*   Test the EMI Eligibility and EMI Amount prediction functionalities.
*   Check for any errors in the Streamlit Cloud logs if issues arise.

**Happy Deploying!**
