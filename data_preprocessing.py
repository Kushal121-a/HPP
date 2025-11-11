import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class HouseDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price'
        
    def load_data(self, filepath):
        """Load house data from CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Please run data_generator.py first.")
            return None
    
    def explore_data(self, df):
        """Basic data exploration and statistics"""
        print("=== DATA EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nPrice statistics:")
        print(f"  Mean: ${df[self.target_column].mean():,.0f}")
        print(f"  Median: ${df[self.target_column].median():,.0f}")
        print(f"  Min: ${df[self.target_column].min():,.0f}")
        print(f"  Max: ${df[self.target_column].max():,.0f}")
        
        return df
    
    def clean_data(self, df):
        """Clean the dataset"""
        print("\n=== DATA CLEANING ===")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"Columns with missing values: {missing_cols}")
            # For numerical columns, fill with median
            for col in missing_cols:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                    print(f"Filled missing values in {col} with median")
        
        # Remove outliers using IQR method for price
        Q1 = df[self.target_column].quantile(0.25)
        Q3 = df[self.target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_rows = len(df)
        df = df[(df[self.target_column] >= lower_bound) & 
                (df[self.target_column] <= upper_bound)]
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} price outliers")
        
        print(f"Final dataset shape: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Create new features and transform existing ones"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create new features
        df['price_per_sqft'] = df[self.target_column] / df['square_feet']
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['age'] = 2024 - df['year_built']
        df['lot_size_acres'] = df['lot_size'] / 43560  # Convert sq ft to acres
        
        # Create interaction features
        df['bedroom_bathroom_ratio'] = df['bedrooms'] / df['bathrooms']
        df['size_efficiency'] = df['square_feet'] * df['energy_efficiency'] / 100
        
        # Create categorical features
        df['price_category'] = pd.cut(df[self.target_column], 
                                    bins=[0, 300000, 500000, 800000, float('inf')],
                                    labels=['Low', 'Medium', 'High', 'Luxury'])
        
        df['size_category'] = pd.cut(df['square_feet'],
                                   bins=[0, 1000, 1500, 2000, float('inf')],
                                   labels=['Small', 'Medium', 'Large', 'Extra Large'])
        
        print("New features created:")
        new_features = ['price_per_sqft', 'total_rooms', 'age', 'lot_size_acres',
                       'bedroom_bathroom_ratio', 'size_efficiency', 'price_category', 'size_category']
        for feature in new_features:
            print(f"  - {feature}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Select numerical features for modeling
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features.remove(self.target_column)
        
        # Remove any infinite or NaN values
        df_clean = df[numerical_features + [self.target_column]].replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        self.feature_columns = numerical_features
        
        print(f"Selected {len(self.feature_columns)} numerical features:")
        for feature in self.feature_columns:
            print(f"  - {feature}")
        
        return df_clean
    
    def scale_features(self, df):
        """Scale numerical features"""
        print("\n=== FEATURE SCALING ===")
        
        if self.feature_columns is None:
            print("Error: Features not prepared. Run prepare_features() first.")
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df[self.feature_columns])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns, index=df.index)
        
        print("Features scaled using StandardScaler")
        return X_scaled_df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n=== DATA SPLITTING ===")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_visualizations(self, df):
        """Create exploratory data analysis visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('House Price Data Analysis', fontsize=16, fontweight='bold')
        
        # Price distribution
        axes[0, 0].hist(df[self.target_column], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('House Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Price vs Square Feet
        axes[0, 1].scatter(df['square_feet'], df[self.target_column], alpha=0.6, color='green')
        axes[0, 1].set_title('Price vs Square Feet')
        axes[0, 1].set_xlabel('Square Feet')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Price vs Bedrooms
        axes[0, 2].boxplot([df[df['bedrooms'] == i][self.target_column] 
                           for i in sorted(df['bedrooms'].unique())], 
                           labels=sorted(df['bedrooms'].unique()))
        axes[0, 2].set_title('Price vs Bedrooms')
        axes[0, 2].set_xlabel('Number of Bedrooms')
        axes[0, 2].set_ylabel('Price ($)')
        
        # Correlation heatmap
        if self.feature_columns is not None:
            corr_matrix = df[self.feature_columns + [self.target_column]].corr()
        else:
            # Use numerical columns if feature_columns not set
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], fmt='.2f', square=True)
        axes[1, 0].set_title('Feature Correlation Matrix')
        
        # Price vs Year Built
        axes[1, 1].scatter(df['year_built'], df[self.target_column], alpha=0.6, color='orange')
        axes[1, 1].set_title('Price vs Year Built')
        axes[1, 1].set_xlabel('Year Built')
        axes[1, 1].set_ylabel('Price ($)')
        
        # Price vs School Rating
        axes[1, 2].scatter(df['school_rating'], df[self.target_column], alpha=0.6, color='red')
        axes[1, 2].set_title('Price vs School Rating')
        axes[1, 2].set_xlabel('School Rating')
        axes[1, 2].set_ylabel('Price ($)')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'data_analysis.png'")
    
    def get_feature_importance_analysis(self, df):
        """Analyze feature importance based on correlation with price"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if self.feature_columns is not None:
            correlations = df[self.feature_columns + [self.target_column]].corr()[self.target_column].abs()
            correlations = correlations.drop(self.target_column).sort_values(ascending=False)
        else:
            # Use numerical columns if feature_columns not set
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in numerical_cols:
                correlations = df[numerical_cols].corr()[self.target_column].abs()
                correlations = correlations.drop(self.target_column).sort_values(ascending=False)
            else:
                print("Target column not found in numerical columns")
                return None
        
        print("Feature importance (correlation with price):")
        for feature, corr in correlations.items():
            print(f"  {feature}: {corr:.4f}")
        
        return correlations

def main():
    """Main function to demonstrate data preprocessing"""
    preprocessor = HouseDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('house_data.csv')
    if df is None:
        return
    
    # Explore data
    df = preprocessor.explore_data(df)
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # Prepare features
    df_clean = preprocessor.prepare_features(df)
    
    # Scale features
    X_scaled = preprocessor.scale_features(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_scaled, df_clean[preprocessor.target_column]
    )
    
    # Create visualizations
    preprocessor.create_visualizations(df)
    
    # Feature importance analysis
    preprocessor.get_feature_importance_analysis(df)
    
    print("\n=== PREPROCESSING COMPLETE ===")
    print("Data is ready for model training!")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main() 