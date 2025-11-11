import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import random

def generate_house_data(n_samples=1000, random_state=42):
    """
    Generate synthetic house price data with realistic features
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Generate base features
    n_features = 15
    
    # Create synthetic data with controlled noise
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          n_informative=10, noise=0.1, random_state=random_state)
    
    # Define feature names
    feature_names = [
        'square_feet', 'bedrooms', 'bathrooms', 'garage_spaces',
        'year_built', 'lot_size', 'distance_to_city_center',
        'school_rating', 'crime_rate', 'walkability_score',
        'public_transport_score', 'parking_score', 'garden_size',
        'energy_efficiency', 'renovation_score'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Make features more realistic
    df['square_feet'] = np.abs(df['square_feet'] * 500 + 800)  # 800-1300 sq ft
    df['bedrooms'] = np.clip(np.round(df['bedrooms'] * 2 + 3), 1, 6)  # 1-6 bedrooms
    df['bathrooms'] = np.clip(np.round(df['bathrooms'] * 1.5 + 2), 1, 4)  # 1-4 bathrooms
    df['garage_spaces'] = np.clip(np.round(df['garage_spaces'] + 1), 0, 3)  # 0-3 spaces
    df['year_built'] = np.clip(np.round(df['year_built'] * 20 + 1990), 1950, 2023)  # 1950-2023
    df['lot_size'] = np.abs(df['lot_size'] * 2000 + 5000)  # 5000-7000 sq ft
    df['distance_to_city_center'] = np.abs(df['distance_to_city_center'] * 5 + 10)  # 10-15 miles
    df['school_rating'] = np.clip(df['school_rating'] * 2 + 7, 1, 10)  # 1-10 rating
    df['crime_rate'] = np.clip(df['crime_rate'] * 0.5 + 0.3, 0.1, 0.8)  # 0.1-0.8 per 1000
    df['walkability_score'] = np.clip(df['walkability_score'] * 20 + 70, 20, 100)  # 20-100
    df['public_transport_score'] = np.clip(df['public_transport_score'] * 20 + 70, 20, 100)  # 20-100
    df['parking_score'] = np.clip(df['parking_score'] * 20 + 70, 20, 100)  # 20-100
    df['garden_size'] = np.abs(df['garden_size'] * 500 + 1000)  # 1000-1500 sq ft
    df['energy_efficiency'] = np.clip(df['energy_efficiency'] * 20 + 70, 30, 100)  # 30-100
    df['renovation_score'] = np.clip(df['renovation_score'] * 20 + 70, 30, 100)  # 30-100
    
    # Generate realistic house prices based on features
    base_price = 200000
    price_multipliers = {
        'square_feet': 100,      # $100 per sq ft
        'bedrooms': 15000,       # $15k per bedroom
        'bathrooms': 25000,      # $25k per bathroom
        'garage_spaces': 10000,  # $10k per garage space
        'year_built': 1000,      # $1k per year (newer = more expensive)
        'lot_size': 5,           # $5 per sq ft of lot
        'school_rating': 5000,   # $5k per rating point
        'energy_efficiency': 1000, # $1k per efficiency point
        'renovation_score': 800   # $800 per renovation point
    }
    
    # Calculate base price
    prices = base_price
    for feature, multiplier in price_multipliers.items():
        if feature in ['year_built']:
            # Newer houses are more expensive
            prices += (df[feature] - 1950) * multiplier
        else:
            prices += df[feature] * multiplier
    
    # Add location premium (distance to city center)
    location_premium = (20 - df['distance_to_city_center']) * 5000
    prices += location_premium
    
    # Add noise and market variation
    prices += np.random.normal(0, 20000, n_samples)
    prices = np.maximum(prices, 50000)  # Minimum price $50k
    
    # Add target column
    df['price'] = prices
    
    return df

def save_data(df, filename='house_data.csv'):
    """Save the generated data to CSV"""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Dataset shape: {df.shape}")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

if __name__ == "__main__":
    # Generate data
    house_data = generate_house_data(n_samples=2000)
    
    # Save data
    save_data(house_data)
    
    # Display sample
    print("\nSample data:")
    print(house_data.head())
    
    # Display statistics
    print("\nData statistics:")
    print(house_data.describe()) 