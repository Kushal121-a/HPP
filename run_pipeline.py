#!/usr/bin/env python3
"""
House Price Prediction - Complete ML Pipeline
This script runs the entire machine learning pipeline from start to finish.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step_num, total_steps, description):
    """Print a formatted step description"""
    print(f"\n📋 Step {step_num}/{total_steps}: {description}")
    print("-" * 50)

def check_file_exists(filename, description):
    """Check if a file exists and print status"""
    if os.path.exists(filename):
        print(f"✅ {description}: {filename}")
        return True
    else:
        print(f"❌ {description}: {filename} (not found)")
        return False

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n🔄 Running {description}...")
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def main():
    """Main pipeline execution"""
    print_header("House Price Prediction - Complete ML Pipeline")
    
    start_time = time.time()
    total_steps = 4
    current_step = 0
    
    # Step 1: Generate Data
    current_step += 1
    print_step(current_step, total_steps, "Data Generation")
    
    if not run_script("data_generator.py", "Data Generation"):
        print("❌ Pipeline failed at data generation step!")
        return False
    
    # Check if data file was created
    if not check_file_exists("house_data.csv", "House dataset"):
        print("❌ Data file not created. Pipeline failed!")
        return False
    
    # Step 2: Data Preprocessing
    current_step += 1
    print_step(current_step, total_steps, "Data Preprocessing")
    
    if not run_script("data_preprocessing.py", "Data Preprocessing"):
        print("❌ Pipeline failed at data preprocessing step!")
        return False
    
    # Step 3: Model Training
    current_step += 1
    print_step(current_step, total_steps, "Model Training")
    
    if not run_script("model_training.py", "Model Training"):
        print("❌ Pipeline failed at model training step!")
        return False
    
    # Check if model file was created
    if not check_file_exists("best_house_price_model.pkl", "Trained model"):
        print("❌ Model file not created. Pipeline failed!")
        return False
    
    # Step 4: Verify Outputs
    current_step += 1
    print_step(current_step, total_steps, "Verification")
    
    # Check all expected output files
    expected_files = [
        ("house_data.csv", "House dataset"),
        ("best_house_price_model.pkl", "Trained model"),
        ("data_analysis.png", "Data analysis plots"),
        ("model_comparison.png", "Model comparison plots"),
        ("predictions_vs_actual.png", "Prediction accuracy plots")
    ]
    
    all_files_exist = True
    for filename, description in expected_files:
        if not check_file_exists(filename, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("⚠️  Some output files are missing. Pipeline may have had issues.")
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Final summary
    print_header("Pipeline Execution Summary")
    
    if all_files_exist:
        print("🎉 SUCCESS: All pipeline steps completed successfully!")
    else:
        print("⚠️  PARTIAL SUCCESS: Pipeline completed with some issues.")
    
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Next steps
    print("\n🚀 Next Steps:")
    print("1. Launch the web application:")
    print("   streamlit run prediction_app.py")
    print("\n2. Or run individual components:")
    print("   python data_generator.py      # Generate new data")
    print("   python data_preprocessing.py  # Preprocess data")
    print("   python model_training.py      # Train models")
    
    return all_files_exist

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("Dependency Check")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'xgboost', 'streamlit', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

if __name__ == "__main__":
    print("🏠 House Price Prediction Pipeline")
    print("This script will run the complete ML pipeline from data generation to model training.")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running the pipeline.")
        sys.exit(1)
    
    # Ask for confirmation
    response = input("\n🤔 Do you want to continue with the pipeline? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = main()
        if success:
            print("\n🎉 Pipeline completed successfully! You can now run the web app.")
        else:
            print("\n❌ Pipeline encountered errors. Please check the output above.")
            sys.exit(1)
    else:
        print("\n👋 Pipeline execution cancelled.")
        print("You can run individual components manually:")
        print("  python data_generator.py")
        print("  python data_preprocessing.py")
        print("  python model_training.py")
        print("  streamlit run prediction_app.py") 