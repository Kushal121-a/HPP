#!/usr/bin/env python3
"""
House Price Prediction - Quick Demo
This script demonstrates the system with a sample prediction without running the full pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_demo_model():
    """Create a simple demo model for demonstration"""
    print("🏠 Creating Demo House Price Prediction Model...")
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic house features
    data = {
        'square_feet': np.random.randint(800, 3000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'garage_spaces': np.random.randint(0, 3, n_samples),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'lot_size': np.random.randint(4000, 12000, n_samples),
        'distance_to_city_center': np.random.randint(5, 25, n_samples),
        'school_rating': np.random.randint(5, 10, n_samples),
        'crime_rate': np.random.uniform(0.1, 0.8, n_samples),
        'walkability_score': np.random.randint(30, 95, n_samples),
        'public_transport_score': np.random.randint(30, 95, n_samples),
        'parking_score': np.random.randint(30, 95, n_samples),
        'garden_size': np.random.randint(500, 2000, n_samples),
        'energy_efficiency': np.random.randint(40, 95, n_samples),
        'renovation_score': np.random.randint(40, 95, n_samples)
    }
    
    # Create derived features
    df = pd.DataFrame(data)
    df['price_per_sqft'] = 150  # Base price per sq ft
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['age'] = 2024 - df['year_built']
    df['lot_size_acres'] = df['lot_size'] / 43560
    df['bedroom_bathroom_ratio'] = df['bedrooms'] / df['bathrooms']
    df['size_efficiency'] = df['square_feet'] * df['energy_efficiency'] / 100
    
    # Generate realistic prices
    base_price = 200000
    price_multipliers = {
        'square_feet': 100,
        'bedrooms': 15000,
        'bathrooms': 25000,
        'garage_spaces': 10000,
        'year_built': 1000,
        'lot_size': 5,
        'school_rating': 5000,
        'energy_efficiency': 1000,
        'renovation_score': 800
    }
    
    prices = base_price
    for feature, multiplier in price_multipliers.items():
        if feature in ['year_built']:
            prices += (df[feature] - 1950) * multiplier
        else:
            prices += df[feature] * multiplier
    
    # Add location premium
    location_premium = (25 - df['distance_to_city_center']) * 3000
    prices += location_premium
    
    # Add noise
    prices += np.random.normal(0, 15000, n_samples)
    prices = np.maximum(prices, 50000)
    
    df['price'] = prices
    
    print(f"✅ Generated {n_samples} sample house records")
    return df

def train_demo_model(df):
    """Train a simple Random Forest model"""
    print("🧠 Training Demo Model...")
    
    # Select features for modeling
    feature_columns = [col for col in df.columns if col != 'price']
    X = df[feature_columns]
    y = df['price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_scaled, y)
    
    print("✅ Model trained successfully!")
    return model, scaler, feature_columns

def make_demo_prediction(model, scaler, feature_columns):
    """Make a sample prediction"""
    print("\n🏠 Making Sample House Price Prediction...")
    
    # Sample house features
    sample_house = {
        'square_feet': 1800,
        'bedrooms': 3,
        'bathrooms': 2,
        'garage_spaces': 2,
        'year_built': 2005,
        'lot_size': 7500,
        'distance_to_city_center': 12,
        'school_rating': 8,
        'crime_rate': 0.3,
        'walkability_score': 75,
        'public_transport_score': 70,
        'parking_score': 80,
        'garden_size': 1200,
        'energy_efficiency': 85,
        'renovation_score': 80,
        'price_per_sqft': 150,
        'total_rooms': 5,
        'age': 19,
        'lot_size_acres': 0.17,
        'bedroom_bathroom_ratio': 1.5,
        'size_efficiency': 1530
    }
    
    # Create feature array
    features = [sample_house[col] for col in feature_columns]
    features_scaled = scaler.transform([features])
    
    # Make prediction
    predicted_price = model.predict(features_scaled)[0]
    
    # Display results
    print("\n" + "="*50)
    print("🏠 SAMPLE HOUSE FEATURES")
    print("="*50)
    
    col1, col2 = [], []
    for i, (feature, value) in enumerate(sample_house.items()):
        if i % 2 == 0:
            col1.append(f"{feature.replace('_', ' ').title()}: {value}")
        else:
            col2.append(f"{feature.replace('_', ' ').title()}: {value}")
    
    # Print in two columns
    max_len = max(len(col1), len(col2))
    for i in range(max_len):
        left = col1[i] if i < len(col1) else ""
        right = col2[i] if i < len(col2) else ""
        print(f"{left:<35} {right}")
    
    print("\n" + "="*50)
    print("💰 PREDICTED HOUSE PRICE")
    print("="*50)
    print(f"Predicted Price: ${predicted_price:,.0f}")
    print(f"Price per sq ft: ${predicted_price/sample_house['square_feet']:.0f}")
    
    # Price analysis
    if predicted_price > 600000:
        category = "Luxury"
        description = "High-end property with premium features"
    elif predicted_price > 400000:
        category = "Premium"
        description = "Quality family home in good location"
    elif predicted_price > 250000:
        category = "Mid-Range"
        description = "Good value property for families"
    else:
        category = "Affordable"
        description = "Great starter home or investment property"
    
    print(f"Category: {category}")
    print(f"Description: {description}")
    
    return predicted_price, sample_house

def show_feature_importance(model, feature_columns):
    """Show feature importance from the model"""
    print("\n" + "="*50)
    print("🔍 FEATURE IMPORTANCE")
    print("="*50)
    
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature'].replace('_', ' ').title():<25} {row['importance']:.4f}")

def main():
    """Main demo function"""
    print("🏠 House Price Prediction - Quick Demo")
    print("="*50)
    print("This demo creates a simple model and makes a sample prediction.")
    print("For the full system, run: python run_pipeline.py")
    print("="*50)
    
    try:
        # Create demo data
        df = create_demo_model()
        
        # Train model
        model, scaler, feature_columns = train_demo_model(df)
        
        # Make prediction
        predicted_price, sample_house = make_demo_prediction(model, scaler, feature_columns)
        
        # Show feature importance
        show_feature_importance(model, feature_columns)
        
        # Summary
        print("\n" + "="*50)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("The demo model predicted a house price of:")
        print(f"${predicted_price:,.0f}")
        print("\nTo run the full system:")
        print("1. python run_pipeline.py     # Complete pipeline")
        print("2. streamlit run prediction_app.py  # Web application")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please check that all required packages are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 