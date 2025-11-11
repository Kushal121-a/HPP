import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePriceModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        
    def train_linear_models(self, X_train, y_train, X_test, y_test):
        """Train linear regression models"""
        print("=== TRAINING LINEAR MODELS ===")
        
        # Linear Regression
        print("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        
        # Ridge Regression
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        
        # Lasso Regression
        print("Training Lasso Regression...")
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        
        # Store models
        self.models['Linear Regression'] = {
            'model': lr,
            'predictions': lr_pred,
            'actual': y_test
        }
        
        self.models['Ridge Regression'] = {
            'model': ridge,
            'predictions': ridge_pred,
            'actual': y_test
        }
        
        self.models['Lasso Regression'] = {
            'model': lasso,
            'predictions': lasso_pred,
            'actual': y_test
        }
        
        print("Linear models training completed!")
    
    def train_tree_models(self, X_train, y_train, X_test, y_test):
        """Train tree-based models"""
        print("\n=== TRAINING TREE-BASED MODELS ===")
        
        # Decision Tree
        print("Training Decision Tree...")
        dt = DecisionTreeRegressor(random_state=42, max_depth=10)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Store models
        self.models['Decision Tree'] = {
            'model': dt,
            'predictions': dt_pred,
            'actual': y_test
        }
        
        self.models['Random Forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'actual': y_test
        }
        
        self.models['Gradient Boosting'] = {
            'model': gb,
            'predictions': gb_pred,
            'actual': y_test
        }
        
        self.models['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'actual': y_test
        }
        
        print("Tree-based models training completed!")
    
    def train_other_models(self, X_train, y_train, X_test, y_test):
        """Train other types of models"""
        print("\n=== TRAINING OTHER MODELS ===")
        
        # Support Vector Regression
        print("Training Support Vector Regression...")
        svr = SVR(kernel='rbf', C=100, gamma='scale')
        svr.fit(X_train, y_train)
        svr_pred = svr.predict(X_test)
        
        # K-Nearest Neighbors
        print("Training K-Nearest Neighbors...")
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        
        # Store models
        self.models['SVR'] = {
            'model': svr,
            'predictions': svr_pred,
            'actual': y_test
        }
        
        self.models['KNN'] = {
            'model': knn,
            'predictions': knn_pred,
            'actual': y_test
        }
        
        print("Other models training completed!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        for name, model_data in self.models.items():
            predictions = model_data['predictions']
            actual = model_data['actual']
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        # Find best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]['model']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"R² Score: {results[best_model_name]['R²']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\n=== HYPERPARAMETER TUNING FOR {model_name.upper()} ===")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
            
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        print("Performing Grid Search...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the best model
        self.best_model = grid_search.best_estimator_
        self.models[f'{model_name} (Tuned)'] = {
            'model': self.best_model,
            'predictions': self.best_model.predict(X_train),  # Will be updated later
            'actual': y_train
        }
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if self.best_model is None:
            print("No best model available. Train models first.")
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            print("Feature importance not available for this model type.")
            return None
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        print("Top 10 most important features:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance_df
    
    def plot_model_comparison(self, results):
        """Plot comparison of all models"""
        print("\n=== CREATING MODEL COMPARISON PLOTS ===")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² Score comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['R²'] for name in model_names]
        
        axes[0, 0].barh(model_names, r2_scores, color='skyblue')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_xlim(0, 1)
        
        # RMSE comparison
        rmse_scores = [results[name]['RMSE'] for name in model_names]
        axes[0, 1].barh(model_names, rmse_scores, color='lightcoral')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_xlabel('RMSE ($)')
        
        # MAE comparison
        mae_scores = [results[name]['MAE'] for name in model_names]
        axes[1, 0].barh(model_names, mae_scores, color='lightgreen')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_xlabel('MAE ($)')
        
        # MAPE comparison
        mape_scores = [results[name]['MAPE'] for name in model_names]
        axes[1, 1].barh(model_names, mape_scores, color='gold')
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_xlabel('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Model comparison plots saved as 'model_comparison.png'")
    
    def plot_predictions_vs_actual(self):
        """Plot predictions vs actual values for the best model"""
        print("\n=== CREATING PREDICTIONS VS ACTUAL PLOT ===")
        
        if self.best_model_name is None:
            print("No best model available.")
            return
        
        model_data = self.models[self.best_model_name]
        predictions = model_data['predictions']
        actual = model_data['actual']
        
        plt.figure(figsize=(10, 8))
        plt.scatter(actual, predictions, alpha=0.6, color='blue')
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Predictions vs Actual - {self.best_model_name}')
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(actual, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Predictions vs actual plot saved as 'predictions_vs_actual.png'")
    
    def save_best_model(self, filename='best_house_price_model.pkl'):
        """Save the best trained model"""
        if self.best_model is not None:
            joblib.dump(self.best_model, filename)
            print(f"\nBest model saved as '{filename}'")
        else:
            print("No best model to save.")
    
    def load_model(self, filename='best_house_price_model.pkl'):
        """Load a saved model"""
        try:
            self.best_model = joblib.load(filename)
            print(f"Model loaded from '{filename}'")
            return self.best_model
        except FileNotFoundError:
            print(f"Model file '{filename}' not found.")
            return None

def main():
    """Main function to demonstrate model training"""
    # Load preprocessed data
    try:
        from data_preprocessing import HouseDataPreprocessor
        
        preprocessor = HouseDataPreprocessor()
        df = preprocessor.load_data('house_data.csv')
        
        if df is None:
            print("Please run data_generator.py first to create the dataset.")
            return
        
        # Preprocess data
        df = preprocessor.clean_data(df)
        df = preprocessor.feature_engineering(df)
        df_clean = preprocessor.prepare_features(df)
        X_scaled = preprocessor.scale_features(df_clean)
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X_scaled, df_clean[preprocessor.target_column]
        )
        
        print("Data loaded and preprocessed successfully!")
        
    except ImportError:
        print("Please run data_preprocessing.py first to set up the data.")
        return
    
    # Initialize trainer
    trainer = HousePriceModelTrainer()
    
    # Train all models
    trainer.train_linear_models(X_train, y_train, X_test, y_test)
    trainer.train_tree_models(X_train, y_train, X_test, y_test)
    trainer.train_other_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    results = trainer.evaluate_models()
    
    # Hyperparameter tuning for the best tree-based model
    if 'Random Forest' in results:
        trainer.hyperparameter_tuning(X_train, y_train, 'Random Forest')
    # Get feature importance
    if hasattr(X_train, 'columns'):
        feature_names = list(X_train.columns)
    else:
        feature_names = [f"feature_{i}" for i in range(len(X_train[0]))]
    trainer.get_feature_importance(feature_names)

    # Create visualizations
    trainer.plot_model_comparison(results)
    trainer.plot_predictions_vs_actual()
    
    # Save the best model
    trainer.save_best_model()
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print("All models have been trained and evaluated!")
    print(f"Best model: {trainer.best_model_name}")
    
    return trainer

if __name__ == "__main__":
    main() 