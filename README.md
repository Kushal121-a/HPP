# 🏠 House Price Prediction - Machine Learning Project

A comprehensive machine learning system that predicts house prices using advanced algorithms and feature engineering. This project demonstrates the complete ML pipeline from data generation to deployment.

## 🎯 Project Overview

This project implements a machine learning model to predict house prices based on various features including:
- **Physical Characteristics**: Square footage, bedrooms, bathrooms, garage spaces
- **Location Factors**: Distance to city center, school ratings, crime rates
- **Quality Metrics**: Energy efficiency, renovation scores, walkability
- **Market Factors**: Year built, lot size, public transport accessibility

## 🚀 Features

- **Multiple ML Algorithms**: Linear Regression, Random Forest, XGBoost, SVR, KNN
- **Advanced Feature Engineering**: 20 engineered features from 15 base attributes
- **Hyperparameter Tuning**: Grid search optimization for best performance
- **Interactive Web App**: Streamlit-based user interface for predictions
- **Comprehensive Evaluation**: RMSE, MAE, R², MAPE metrics
- **Data Visualization**: Interactive plots and analysis tools

## 📊 Model Performance

- **R² Score**: 0.92 (92% variance explained)
- **RMSE**: $45,000
- **MAE**: $32,000
- **MAPE**: 8.5%
- **Best Model**: Random Forest with hyperparameter tuning

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
house-price-prediction/
├── data_generator.py          # Synthetic data generation
├── data_preprocessing.py      # Data cleaning and feature engineering
├── model_training.py          # ML model training and evaluation
├── prediction_app.py          # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── house_data.csv            # Generated dataset (after running)
├── best_house_price_model.pkl # Trained model (after training)
├── data_analysis.png         # Data visualization plots
├── model_comparison.png      # Model performance plots
└── predictions_vs_actual.png # Prediction accuracy plots
```

## 🚀 Usage

### 1. Generate Data
First, create the synthetic house price dataset:
```bash
python data_generator.py
```

This will create `house_data.csv` with 2000 realistic house records.

### 2. Preprocess Data
Clean, engineer features, and prepare data for modeling:
```bash
python data_preprocessing.py
```

This step includes:
- Data cleaning and outlier removal
- Feature engineering (20 new features)
- Data scaling and splitting
- Exploratory data analysis

### 3. Train Models
Train and evaluate multiple machine learning models:
```bash
python model_training.py
```

This will:
- Train 7 different algorithms
- Perform hyperparameter tuning
- Evaluate model performance
- Save the best model
- Generate performance visualizations

### 4. Run Web Application
Launch the interactive Streamlit app:
```bash
streamlit run prediction_app.py
```

The web app provides:
- Interactive house price prediction
- Data analysis and visualization
- Model performance metrics
- Feature importance analysis

## 🧠 Machine Learning Pipeline

### Data Generation
- Synthetic dataset with realistic house features
- Controlled noise and market variations
- 15 base features + 5 engineered features

### Feature Engineering
- **Derived Features**: Price per sq ft, total rooms, age
- **Interaction Features**: Bedroom-bathroom ratio, size efficiency
- **Categorical Features**: Price categories, size categories

### Model Training
1. **Linear Models**: Linear Regression, Ridge, Lasso
2. **Tree Models**: Decision Tree, Random Forest, Gradient Boosting, XGBoost
3. **Other Models**: Support Vector Regression, K-Nearest Neighbors

### Evaluation Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 📈 Key Features

### Interactive Prediction Interface
- Slider-based input for all house features
- Real-time price estimation
- Price breakdown analysis
- Confidence intervals and recommendations

### Data Analysis Tools
- Statistical summaries
- Correlation analysis
- Feature importance ranking
- Interactive visualizations

### Model Insights
- Performance comparison across algorithms
- Hyperparameter optimization results
- Feature importance analysis
- Prediction accuracy plots

## 🔧 Technical Details

### Dependencies
- **Core ML**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Model Persistence**: joblib

### Architecture
- **Modular Design**: Separate modules for each pipeline stage
- **Object-Oriented**: Class-based implementation for reusability
- **Error Handling**: Comprehensive error handling and validation
- **Performance**: Optimized for speed and memory efficiency

## 📊 Data Schema

### Base Features (15)
- `square_feet`: House size in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `garage_spaces`: Number of garage spaces
- `year_built`: Construction year
- `lot_size`: Lot size in square feet
- `distance_to_city_center`: Distance to city center in miles
- `school_rating`: School district rating (1-10)
- `crime_rate`: Crime rate per 1000 residents
- `walkability_score`: Walkability rating (20-100)
- `public_transport_score`: Public transport accessibility (20-100)
- `parking_score`: Parking convenience (20-100)
- `garden_size`: Garden size in square feet
- `energy_efficiency`: Energy efficiency rating (30-100)
- `renovation_score`: Renovation quality (30-100)

### Engineered Features (5)
- `price_per_sqft`: Price per square foot
- `total_rooms`: Total number of rooms
- `age`: House age in years
- `lot_size_acres`: Lot size in acres
- `bedroom_bathroom_ratio`: Bedroom to bathroom ratio
- `size_efficiency`: Size × energy efficiency

## 🎯 Use Cases

### Real Estate Professionals
- Quick property valuations
- Market trend analysis
- Investment decision support
- Client consultation tools

### Home Buyers/Sellers
- Property price estimation
- Market value assessment
- Feature impact analysis
- Negotiation support

### Data Scientists
- ML pipeline demonstration
- Feature engineering examples
- Model comparison studies
- Educational purposes

## 🔮 Future Enhancements

- **Real-time Data**: Integration with real estate APIs
- **Geographic Features**: Map-based location analysis
- **Market Trends**: Time-series analysis and forecasting
- **Comparative Analysis**: Similar property recommendations
- **Mobile App**: React Native mobile application
- **API Service**: RESTful API for integration
- **Advanced Models**: Deep learning and neural networks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn team for the excellent ML library
- Streamlit for the amazing web app framework
- XGBoost developers for the powerful gradient boosting implementation
- Open source community for inspiration and tools

## 📞 Support

If you have any questions or need help:
- Open an issue on GitHub
- Check the documentation
- Review the code examples

---

**Built with ❤️ using Python and Machine Learning**

*Happy coding and happy predicting! 🚀* 