@echo off
echo.
echo ========================================
echo 🏠 House Price Prediction Pipeline
echo ========================================
echo.
echo This will run the complete ML pipeline:
echo 1. Generate synthetic house data
echo 2. Preprocess and engineer features
echo 3. Train multiple ML models
echo 4. Launch the web application
echo.
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if requirements are installed
echo 🔍 Checking dependencies...
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, plotly, xgboost, streamlit, joblib" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some packages are missing. Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
    echo ✅ Requirements installed successfully
) else (
    echo ✅ All required packages are installed
)

echo.
echo ========================================
echo 🚀 Starting Pipeline Execution
echo ========================================
echo.

REM Run the complete pipeline
echo 📋 Step 1/4: Data Generation
python data_generator.py
if errorlevel 1 (
    echo ❌ Data generation failed
    pause
    exit /b 1
)

echo.
echo 📋 Step 2/4: Data Preprocessing
python data_preprocessing.py
if errorlevel 1 (
    echo ❌ Data preprocessing failed
    pause
    exit /b 1
)

echo.
echo 📋 Step 3/4: Model Training
python model_training.py
if errorlevel 1 (
    echo ❌ Model training failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🎉 Pipeline Completed Successfully!
echo ========================================
echo.
echo 🚀 Launching Web Application...
echo.
echo The web app will open in your browser.
echo To stop the app, press Ctrl+C in this window.
echo.

REM Launch the Streamlit app
streamlit run prediction_app.py

echo.
echo 👋 Web application closed.
echo.
pause 