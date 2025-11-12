
# MLflow Configuration for EMI Prediction System
EXPERIMENT_NAME = "EMI_Prediction_System"
TRACKING_URI = "./mlruns"

# Registered Models
CLASSIFICATION_MODEL = "EMI_Eligibility_Classifier"
REGRESSION_MODEL = "Max_EMI_Regressor"

# Best Model Performance
CLASSIFICATION_ACCURACY = 0.9771
REGRESSION_R2 = 0.9963

# Feature Information
FEATURE_COUNT = 55
TARGET_CLASSES = ['Eligible', 'High_Risk', 'Not_Eligible']
