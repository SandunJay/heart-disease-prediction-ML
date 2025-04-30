<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Heart Disease Prediction Using Machine Learning

A comprehensive machine learning project that implements and compares Random Forest, XGBoost, and SVM models for heart disease prediction with detailed performance analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation Instructions](#installation-instructions)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Results and Evaluation](#results-and-evaluation)
- [Usage Examples](#usage-examples)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)


## Project Overview

This project aims to build and compare machine learning models for predicting heart disease based on patient data. Heart disease remains one of the leading causes of mortality worldwide, and early detection through machine learning techniques can significantly improve patient outcomes.

The project implements three popular machine learning algorithms:

- Random Forest
- XGBoost
- Support Vector Machine (SVM)

Each model is thoroughly evaluated and compared to determine the most effective approach for heart disease prediction.

## Dataset Description

The dataset used in this project contains various features related to heart health:

- **Demographic features**: Age, Sex
- **Clinical measurements**: RestingBP, Cholesterol, FastingBS, MaxHR
- **Symptoms**: ChestPainType, RestingECG, ExerciseAngina
- **Test results**: ST_Slope, Oldpeak

The dataset was preprocessed to remove physiologically impossible values (Cholesterol = 0) and handle categorical features through one-hot encoding.

## Installation Instructions

```bash
# Clone this repository
git clone https://github.com/yourusername/heart-disease-prediction.git

# Navigate to the project directory
cd heart-disease-prediction

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```


## Project Structure

```
heart-disease-prediction/
│
├── data/
│   └── heart.csv                # Heart disease dataset
│
├── notebooks/
│   ├── data_exploration.ipynb   # Data exploration and visualization
│   ├── random_forest.ipynb      # Random Forest implementation
│   ├── xgboost.ipynb            # XGBoost implementation
│   ├── svm.ipynb                # SVM implementation
│   └── model_comparison.ipynb   # Model comparison and analysis
│
├── src/
│   ├── preprocess.py            # Data preprocessing functions
│   ├── feature_engineering.py   # Feature engineering functions
│   ├── models.py                # Model implementation functions
│   ├── evaluation.py            # Model evaluation functions
│   └── visualization.py         # Result visualization functions
│
├── results/
│   ├── figures/                 # Generated plots and figures
│   └── models/                  # Saved model files
│
├── requirements.txt             # Project dependencies
├── setup.py                     # Package setup file
└── README.md                    # Project documentation
```


## Models Implemented

### Random Forest

- Ensemble learning method using multiple decision trees
- Hyperparameter optimization for n_estimators, max_depth, min_samples_split
- Feature importance analysis to identify key predictors


### XGBoost

- Gradient boosting implementation optimized for performance
- Hyperparameter tuning for learning_rate, max_depth, subsample
- Early stopping to prevent overfitting


### Support Vector Machine (SVM)

- Implemented with multiple kernel functions (linear, polynomial, RBF)
- Regularization parameter (C) optimization
- Gamma parameter tuning for non-linear kernels


## Results and Evaluation

All models were evaluated using multiple performance metrics:


| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Random Forest | 0.8800 | 0.8649 | 0.8889 | 0.8767 | 0.9374 |
| XGBoost | 0.8200 | 0.8169 | 0.8056 | 0.8112 | 0.9097 |
| SVM | 0.8467 | 0.8182 | 0.8750 | 0.8456 | 0.9257 |

**Key Findings:**

- Random Forest achieved the highest performance across most metrics
- Its superior recall (0.8889) is particularly important for healthcare applications
- Cross-validation confirmed consistent performance across different data subsets
- Feature importance analysis revealed that Oldpeak, ST_Slope, and chest pain type were the strongest predictors


## Usage Examples

```python
# Load preprocessed data
from src.preprocess import load_and_preprocess_data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/heart.csv')

# Train and evaluate Random Forest model
from src.models import train_random_forest
from src.evaluation import evaluate_model
rf_model = train_random_forest(X_train, y_train)
rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

# Make predictions on new data
new_patient_data = [...] # Input features for a new patient
prediction = rf_model.predict(new_patient_data)
probability = rf_model.predict_proba(new_patient_data)[:, 1]
```


## Future Work

- Integrate **Explainable AI (XAI)** techniques for better model interpretability
- Implement **feature selection methods** to identify optimal predictor subsets
- Explore **ensemble methods** combining the strengths of multiple models
- Develop a **web application** for interactive heart disease risk assessment
- Investigate **deep learning approaches** for improved predictive performance
- Expand the dataset with additional clinical features to enhance prediction accuracy


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- Heart Disease Dataset: UCI Machine Learning Repository
- Scikit-learn Documentation
- XGBoost Documentation
- Research papers on heart disease prediction using machine learning


## License

This project is licensed under the MIT License - see the LICENSE file for details.

