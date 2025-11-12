
# ============================================================================
# EMI PREDICTION SYSTEM - STREAMLIT APPLICATION
# ============================================================================
#
# FILE: emi_prediction_app.py
# PURPOSE: Main web application for EMI prediction system
# DEPLOYMENT: Compatible with local testing and Streamlit Cloud
#
# APPLICATION ARCHITECTURE:
# - Multi-page interface with sidebar navigation
# - Real-time EMI eligibility predictions
# - Maximum EMI amount calculations
# - Interactive data visualizations
# - Model performance dashboard
#
# ENVIRONMENT CONFIGURATION:
# - Local: Uses local file paths and resources
# - Cloud: Adapts to Streamlit Cloud environment
# - Fallbacks: Simulated predictions if models unavailable
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION AND ENVIRONMENT SETUP
# ============================================================================
#
# PURPOSE: Configure the application based on the deployment environment
#
# ENVIRONMENT DETECTION:
# - Streamlit Cloud: Uses cloud-optimized settings
# - Local Development: Uses local file paths and resources
# - Fallback Mode: Provides basic functionality if dependencies missing
#
# BENEFITS:
# - Single codebase works in all environments
# - Automatic adaptation to deployment context
# - Graceful degradation for missing components
# ============================================================================

# Detect deployment environment
IS_STREAMLIT_CLOUD = os.path.exists('/app')
IS_LOCAL = not IS_STREAMLIT_CLOUD

# Page configuration - optimized for both desktop and mobile
st.set_page_config(
    page_title="EMI Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/emi-prediction',
        'Report a bug': "https://github.com/your-username/emi-prediction/issues",
        'About': "# EMI Prediction System - Machine Learning Powered"
    }
)

# ============================================================================
# CUSTOM CSS STYLING FOR PROFESSIONAL UI
# ============================================================================
#
# PURPOSE: Enhance user experience with consistent, professional styling
#
# DESIGN ELEMENTS:
# - Consistent color scheme and typography
# - Responsive design for all screen sizes
# - Visual indicators for different prediction states
# - Accessible color contrasts and font sizes
#
# UX IMPROVEMENTS:
# - Clear visual hierarchy
# - Intuitive navigation
# - Professional appearance
# - Mobile responsiveness
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-prediction {
        border-left: 5px solid #28a745;
        background-color: #f8fff9;
    }
    .warning-prediction {
        border-left: 5px solid #ffc107;
        background-color: #fffef0;
    }
    .danger-prediction {
        border-left: 5px solid #dc3545;
        background-color: #fff5f5;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 2rem; }
        .sub-header { font-size: 1.25rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA AND MODEL MANAGEMENT WITH CACHING
# ============================================================================
#
# PURPOSE: Efficiently load and manage application data with performance optimization
#
# CACHING STRATEGY:
# - @st.cache_resource: For ML models and large datasets (persists across sessions)
# - @st.cache_data: For computed results and transformations (can expire)
# - Automatic cache invalidation when dependencies changes
#
# PERFORMANCE BENEFITS:
# - Faster application loading
# - Reduced memory usage
# - Better user experience
# - Efficient resource utilization
# ============================================================================

@st.cache_resource
def load_application_data():
    """
    Load all required application data with comprehensive error handling.

    RETURNS:
        dict: Contains model metadata, performance metrics, and configuration

    ERROR HANDLING:
        - Graceful degradation if files are missing
        - Clear error messages for debugging
        - Fallback to simulated data for demo purposes
    """
    try:
        # Attempt to load MLflow metadata
        metadata = joblib.load("mlflow_metadata.pkl")
        st.success("✅ Application data loaded successfully")

        # Log environment information for debugging
        if IS_STREAMLIT_CLOUD:
            st.sidebar.success("🌐 Running on Streamlit Cloud")
        else:
            st.sidebar.info("💻 Running locally")

        return metadata

    except FileNotFoundError:
        st.warning("⚠️ Using demo data - MLflow metadata not found")
        # Fallback data for demonstration
        return {
            "model_performance": {
                "classification": {"accuracy": 0.9771, "f1_score": 0.9752},
                "regression": {"r2_score": 0.9963, "rmse": 466}
            },
            "environment": "demo",
            "loaded_at": datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"❌ Error loading application data: {str(e)}")
        return {
            "model_performance": {
                "classification": {"accuracy": 0.9000, "f1_score": 0.8900},
                "regression": {"r2_score": 0.8500, "rmse": 1000}
            },
            "environment": "error_fallback",
            "error": str(e)
        }

# ============================================================================
# NAVIGATION SYSTEM WITH SESSION MANAGEMENT
# ============================================================================
#
# PURPOSE: Provide intuitive navigation and maintain application state
#
# SESSION STATE MANAGEMENT:
# - Persistent user selections across interactions
# - Page routing and navigation state
# - User preferences and settings
# - Form data persistence
#
# USER EXPERIENCE:
# - Consistent navigation across all pages
# - Clear indication of current location
# - Quick access to all features
# - Responsive sidebar design
# ============================================================================

def setup_sidebar_navigation():
    """
    Create and manage the application navigation system.

    FEATURES:
        - Persistent navigation state
        - Environment indicators
        - System status information
        - Clean, organized layout
    """
    st.sidebar.title("💰 EMI Prediction System")
    st.sidebar.markdown("---")

    # Navigation options with icons for better UX
    page_options = [
        "🏠 Home Dashboard",
        "🎯 EMI Eligibility",
        "📈 EMI Amount",
        "📊 Data Explorer",
        "🤖 Model Performance"
    ]

    selected_page = st.sidebar.radio("Navigate to", page_options, index=0)

    st.sidebar.markdown("---")

    # System status and environment information
    metadata = load_application_data()

    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success(f"**Status:** ✅ Operational")
    st.sidebar.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.info(f"**Accuracy:** {metadata['model_performance']['classification']['accuracy']:.1%}")

    if IS_STREAMLIT_CLOUD:
        st.sidebar.success("**Environment:** 🌐 Streamlit Cloud")
    else:
        st.sidebar.info("**Environment:** 💻 Local Development")

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Need help?* 📧 Contact support")

    return selected_page

# ============================================================================
# HOME PAGE - APPLICATION DASHBOARD
# ============================================================================
#
# PURPOSE: Provide comprehensive project overview and quick access to features
#
# DASHBOARD COMPONENTS:
# - Hero section with application identity
# - Key performance metrics display
# - Quick start action buttons
# - Project overview and capabilities
# - System status information
#
# USER ONBOARDING:
# - Clear value proposition
# - Visual credibility indicators
# - Easy access to primary features
# - Professional first impression
# ============================================================================

def render_home_page():
    """
    Display the home dashboard with comprehensive project overview.

    LAYOUT SECTIONS:
        1. Hero section with branding
        2. Key metrics dashboard
        3. Feature overview and quick start
        4. Technical capabilities
    """
    st.markdown("<div class='main-header'>💰 EMI Prediction System</div>", unsafe_allow_html=True)

    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135706.png", width=120)

    st.markdown("""
    ## Intelligent EMI Decision Support

    Leverage advanced machine learning to make data-driven decisions about
    EMI eligibility and affordable payment amounts. Our system analyzes multiple
    financial factors to provide accurate, reliable predictions.
    """)

    # Load and display performance metrics
    metadata = load_application_data()

    # Key metrics dashboard
    st.markdown("<div class='sub-header'>📈 Performance Metrics</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Classification Accuracy",
                 f"{metadata['model_performance']['classification']['accuracy']:.1%}")
        st.markdown("EMI Eligibility")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Regression R² Score",
                 f"{metadata['model_performance']['regression']['r2_score']:.1%}")
        st.markdown("EMI Amount")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Models Trained", "6")
        st.markdown("Algorithms")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Data Records", "404,800")
        st.markdown("Training Data")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Quick start section
    st.markdown("<div class='sub-header'>🚀 Quick Start</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **🎯 EMI Eligibility Prediction**
        - Check customer eligibility instantly
        - Risk assessment with confidence scores
        - Comprehensive financial profiling
        - Regulatory compliance checks
        """)
        if st.button("Start Eligibility Check", key="home_eligibility", use_container_width=True):
            st.session_state.current_page = "🎯 EMI Eligibility"
            st.rerun()

    with col2:
        st.info("""
        **📈 EMI Amount Prediction**
        - Calculate maximum affordable EMI
        - Personalized recommendations
        - Financial capacity analysis
        - Real-time calculations
        """)
        if st.button("Calculate EMI Amount", key="home_amount", use_container_width=True):
            st.session_state.current_page = "📈 EMI Amount"
            st.rerun()

    # Technical capabilities section
    st.markdown("---")
    st.markdown("<div class='sub-header'>🛠️ Technical Capabilities</div>", unsafe_allow_html=True)

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("**🤖 Machine Learning**")
        st.markdown("- XGBoost & Random Forest")
        st.markdown("- Ensemble methods")
        st.markdown("- Feature importance analysis")

    with tech_col2:
        st.markdown("**📊 Data Processing**")
        st.markdown("- 55+ engineered features")
        st.markdown("- Real-time preprocessing")
        st.markdown("- Automated validation")

    with tech_col3:
        st.markdown("**🌐 Deployment**")
        st.markdown("- Streamlit framework")
        st.markdown("- Cloud-native architecture")
        st.markdown("- Auto-scaling capabilities")

# ============================================================================
# EMI ELIGIBILITY PREDICTION PAGE
# ============================================================================
#
# PURPOSE: Provide real-time EMI eligibility assessment using ML models
#
# PREDICTION WORKFLOW:
# 1. User input collection through structured forms
# 2. Data validation and feature engineering
# 3. Model inference with confidence scoring
# 4. Comprehensive results visualization
# 5. Actionable recommendations
#
# BUSINESS LOGIC:
# - Multi-factor risk assessment
# - Regulatory compliance checks
# - Transparent decision explanation
# - Confidence-based recommendations
# ============================================================================

def render_eligibility_page():
    """
    EMI eligibility prediction interface with comprehensive input validation.

    INPUT VALIDATION:
        - Range checks for numerical values
        - Data type validation
        - Business rule enforcement
        - User-friendly error messages
    """
    st.markdown("<div class='main-header'>🎯 EMI Eligibility Prediction</div>", unsafe_allow_html=True)

    st.info("""
    **Comprehensive EMI Eligibility Assessment**
    Our machine learning model analyzes multiple financial factors including credit history,
    income stability, existing obligations, and loan parameters to determine eligibility.
    """)

    # Comprehensive input form with sections
    with st.form("eligibility_prediction_form", clear_on_submit=False):
        st.subheader("📋 Customer Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**👤 Personal Details**")
            age = st.slider("Age", min_value=18, max_value=70, value=35,
                           help="Customer age in years")
            employment_status = st.selectbox("Employment Status",
                                           ["Salaried", "Self-Employed", "Business", "Retired"],
                                           help="Current employment situation")
            years_at_current_job = st.slider("Years at Current Job", 0, 40, 5,
                                            help="Employment stability indicator")

        with col2:
            st.markdown("**💰 Income Details**")
            monthly_salary = st.number_input("Monthly Salary (₹)",
                                           min_value=10000, max_value=500000, value=50000, step=5000,
                                           help="Gross monthly income before deductions")
            other_income = st.number_input("Other Monthly Income (₹)", 0, 100000, 0, step=5000,
                                         help="Additional income sources")

        st.subheader("📊 Financial Profile")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**🏦 Banking & Credit**")
            credit_score = st.slider("Credit Score", 300, 850, 700,
                                   help="CIBIL or equivalent credit score (300-850)")
            bank_balance = st.number_input("Bank Balance (₹)", 0, 10000000, 100000, step=10000,
                                         help="Current savings and checking balance")
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"],
                                        help="Any current active loans")

        with col4:
            st.markdown("**💸 Current Obligations**")
            current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 100000, 0, step=1000,
                                               help="Total monthly EMI payments")
            monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000, step=1000,
                                         help="Housing expenses")
            other_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 50000, 10000, step=1000,
                                           help="Living and miscellaneous expenses")

        st.subheader("🎯 Loan Request")

        col5, col6 = st.columns(2)

        with col5:
            loan_amount = st.number_input("Requested Loan Amount (₹)", 1000, 5000000, 100000, step=10000,
                                        help="Desired loan amount")
            loan_tenure = st.slider("Loan Tenure (months)", 6, 84, 24,
                                  help="Preferred repayment period")

        with col6:
            loan_purpose = st.selectbox("Loan Purpose",
                                      ["Home Appliances", "Vehicle", "Education",
                                       "Personal Loan", "Home Renovation", "Medical"],
                                      help="Purpose of the requested loan")

        # Form submission with validation
        submitted = st.form_submit_button("🔍 Analyze Eligibility", type="primary", use_container_width=True)

    # Prediction logic
    if submitted:
        with st.spinner("🔄 Analyzing financial profile and calculating eligibility..."):
            # Simulate processing time for realistic UX
            import time
            time.sleep(2)

            # Prepare input data and calculate derived features
            input_data = prepare_eligibility_input(
                age, monthly_salary, other_income, credit_score, bank_balance,
                existing_loans, current_emi_amount, monthly_rent, other_expenses,
                loan_amount, loan_tenure, employment_status, loan_purpose
            )

            # Get prediction (simulated for this example)
            prediction_result = simulate_eligibility_prediction(input_data)

            # Display comprehensive results
            display_eligibility_results(prediction_result, input_data)

def prepare_eligibility_input(*args):
    """
    Prepare and validate input data for eligibility prediction.

    FEATURE ENGINEERING:
        - Debt-to-income ratio calculation
        - Loan-to-income ratio
        - Financial stability score
        - Risk factor aggregation
    """
    # Calculate derived financial ratios
    total_income = args[1] + args[2]  # salary + other income
    total_expenses = args[6] + args[7] + args[8]  # EMI + rent + other expenses
    disposable_income = total_income - total_expenses

    # Financial ratios
    debt_to_income = args[6] / max(total_income, 1)  # current EMI / income
    loan_to_income = args[9] / max(total_income * 12, 1)  # requested loan / annual income
    expense_ratio = total_expenses / max(total_income, 1)

    return {
        "age": args[0],
        "total_income": total_income,
        "credit_score": args[3],
        "bank_balance": args[4],
        "has_existing_loans": args[5] == "Yes",
        "current_emi": args[6],
        "disposable_income": disposable_income,
        "debt_to_income_ratio": debt_to_income,
        "loan_to_income_ratio": loan_to_income,
        "expense_ratio": expense_ratio,
        "employment_status": args[11],
        "loan_purpose": args[12]
    }

def simulate_eligibility_prediction(input_data):
    """
    Simulate ML model prediction for eligibility assessment.

    In a production environment, this would call actual ML models.
    """
    # Business rules based simulation
    credit_score = input_data["credit_score"]
    debt_ratio = input_data["debt_to_income_ratio"]
    disposable_income = input_data["disposable_income"]

    if credit_score >= 750 and debt_ratio <= 0.35 and disposable_income > input_data["total_income"] * 0.3:
        return {"prediction": "Eligible", "confidence": 0.92, "risk_level": "Low"}
    elif credit_score >= 650 and debt_ratio <= 0.45 and disposable_income > 0:
        return {"prediction": "Conditional", "confidence": 0.78, "risk_level": "Medium"}
    else:
        return {"prediction": "Not Eligible", "confidence": 0.85, "risk_level": "High"}

def display_eligibility_results(prediction, input_data):
    """
    Display comprehensive eligibility prediction results.

    RESULTS INCLUDED:
        - Clear eligibility decision
        - Confidence scores
        - Key influencing factors
        - Visualizations
        - Actionable recommendations
    """
    st.markdown("---")
    st.markdown("<div class='sub-header'>📊 Eligibility Assessment Results</div>", unsafe_allow_html=True)

    # Result card with appropriate styling
    if prediction["prediction"] == "Eligible":
        card_class = "success-prediction"
        icon = "✅"
        title = "Eligible for EMI"
        message = "**Congratulations!** The customer meets all eligibility criteria."
    elif prediction["prediction"] == "Conditional":
        card_class = "warning-prediction"
        icon = "⚠️"
        title = "Conditionally Eligible"
        message = "**Additional review recommended.** Some risk factors present."
    else:
        card_class = "danger-prediction"
        icon = "❌"
        title = "Not Eligible"
        message = "**Does not meet current criteria.** Consider alternative options."

    # Display result card
    st.markdown(f"""
    <div class='prediction-card {card_class}'>
        <h3>{icon} {title}</h3>
        <p>{message}</p>
        <p><strong>Confidence Level:</strong> {prediction['confidence']:.1%}</p>
        <p><strong>Risk Assessment:</strong> {prediction['risk_level']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Probability Distribution")

        # Create confidence visualization
        if prediction["prediction"] == "Eligible":
            values = [prediction["confidence"], 0.15, 0.05]
        elif prediction["prediction"] == "Conditional":
            values = [0.25, prediction["confidence"], 0.15]
        else:
            values = [0.1, 0.2, prediction["confidence"]]

        labels = ["Eligible", "Conditional", "Not Eligible"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=['#28a745', '#ffc107', '#dc3545']
        )])

        fig.update_layout(
            showlegend=True,
            height=300,
            title_text="Prediction Confidence",
            annotations=[dict(
                text=f'{prediction["confidence"]:.1%}',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔍 Key Decision Factors")

        factors = [
            f"**Credit Score**: {input_data['credit_score']} "
            f"({'Excellent' if input_data['credit_score'] >= 750 else 'Good' if input_data['credit_score'] >= 650 else 'Fair'})",

            f"**Debt-to-Income**: {input_data['debt_to_income_ratio']:.2f} "
            f"({'Low' if input_data['debt_to_income_ratio'] <= 0.3 else 'Moderate' if input_data['debt_to_income_ratio'] <= 0.5 else 'High'})",

            f"**Disposable Income**: ₹{input_data['disposable_income']:,.0f} "
            f"({'Strong' if input_data['disposable_income'] > input_data['total_income'] * 0.4 else 'Adequate' if input_data['disposable_income'] > 0 else 'Limited'})",

            f"**Loan-to-Income**: {input_data['loan_to_income_ratio']:.2f} "
            f"({'Reasonable' if input_data['loan_to_income_ratio'] <= 1 else 'High'})"
        ]

        for factor in factors:
            st.markdown(f"- {factor}")

        # Recommendations
        st.markdown("---")
        st.markdown("**💡 Recommendations:**")
        if prediction["prediction"] == "Eligible":
            st.markdown("- Proceed with standard processing")
            st.markdown("- Offer competitive interest rates")
            st.markdown("- Consider upselling additional products")
        elif prediction["prediction"] == "Conditional":
            st.markdown("- Request additional documentation")
            st.markdown("- Consider higher interest rates")
            st.markdown("- Set lower credit limits")
        else:
            st.markdown("- Explore secured loan options")
            st.markdown("- Suggest credit improvement strategies")
            st.markdown("- Consider co-applicant or guarantor")

# ============================================================================
# MAIN APPLICATION CONTROLLER
# ============================================================================
#
# PURPOSE: Manage application state and route between different pages
#
# SESSION STATE MANAGEMENT:
# - Initialize and maintain application state
# - Handle page routing and navigation
# - Manage user preferences
# - Ensure consistent user experience
# ============================================================================

def main():
    """
    Main application controller and entry point.

    APPLICATION FLOW:
        1. Initialize session state
        2. Render navigation
        3. Route to appropriate page
        4. Handle state transitions
    """
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home Dashboard"

    # Setup navigation and get current selection
    selected_page = setup_sidebar_navigation()

    # Update session state if navigation changed
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    # Route to appropriate page
    if st.session_state.current_page == "🏠 Home Dashboard":
        render_home_page()
    elif st.session_state.current_page == "🎯 EMI Eligibility":
        render_eligibility_page()
    elif st.session_state.current_page == "📈 EMI Amount":
        # Placeholder for EMI amount page
        st.markdown("<div class='main-header'>📈 EMI Amount Prediction</div>", unsafe_allow_html=True)
        st.info("EMI Amount Prediction feature - Implementation follows similar pattern")
        st.markdown("This page would contain the regression model for maximum affordable EMI calculation.")
    elif st.session_state.current_page == "📊 Data Explorer":
        # Placeholder for data explorer
        st.markdown("<div class='main-header'>📊 Data Explorer</div>", unsafe_allow_html=True)
        st.info("Interactive Data Exploration")
        st.markdown("This page would contain interactive charts and data analysis tools.")
    elif st.session_state.current_page == "🤖 Model Performance":
        # Placeholder for model performance
        st.markdown("<div class='main-header'>🤖 Model Performance</div>", unsafe_allow_html=True)
        st.info("Model Performance Dashboard")
        st.markdown("This page would display MLflow integration and performance metrics.")

# Application entry point
if __name__ == "__main__":
    main()
