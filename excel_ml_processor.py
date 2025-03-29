import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import warnings
warnings.filterwarnings('ignore')


def excel_ml_processor(file_path, target_column=None, problem_type='classification', 
                     test_size=0.2, random_state=42, balance_method='auto',
                     save_model_path='best_model.pkl'):
    """
    Automated ML processor for Excel files that cleans data, handles missing values,
    balances classes if needed, and compares different algorithms to find the best one.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file (.xls, .xlsx)
    target_column : str, optional
        Name of the target column. If None, will try to detect automatically.
    problem_type : str, default='classification'
        Type of ML problem: 'classification' or 'regression'
    test_size : float, default=0.2
        Proportion of the dataset to be used as test set
    random_state : int, default=42
        Random seed for reproducibility
    balance_method : str, default='auto'
        Method to balance data: 'smote', 'undersample', 'auto' (choose based on dataset size), or None
    save_model_path : str, default='best_model.pkl'
        Path to save the best model
        
    Returns:
    --------
    dict
        Dictionary containing best model, metrics, and prediction function
    """
    # Step 1: Load the data
    print(f"Loading data from {file_path}")
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading the Excel file: {e}")
        return None
    
    # Display basic info
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Step 2: Detect target column if not provided
    if target_column is None:
        # Common target column names
        common_targets = ['target', 'label', 'class', 'outcome', 'y', 'response', 
                         'diagnosis', 'disease', 'status', 'result']
        
        for col in common_targets:
            if col in data.columns or col.lower() in [c.lower() for c in data.columns]:
                # Find the actual column name with case insensitivity
                for c in data.columns:
                    if c.lower() == col.lower():
                        target_column = c
                        break
                print(f"Detected target column: {target_column}")
                break
        
        # If still not found, look for binary columns
        if target_column is None:
            for col in data.columns:
                if data[col].nunique() == 2:
                    target_column = col
                    print(f"Using binary column as target: {target_column}")
                    break
    
    # If still no target column
    if target_column is None:
        print("Could not automatically detect target column.")
        print("Available columns:", data.columns.tolist())
        print("Please specify the target column manually.")
        return None
    
    # Check if target column exists
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset")
        return None
    
    print(f"\nUsing '{target_column}' as the target column")
    
    # Step 3: Data Cleaning and Preprocessing
    # Remove duplicates
    initial_rows = data.shape[0]
    data = data.drop_duplicates()
    if initial_rows > data.shape[0]:
        print(f"Removed {initial_rows - data.shape[0]} duplicate rows")
    
    # Handle missing values in target column
    if data[target_column].isnull().sum() > 0:
        print(f"Dropping {data[target_column].isnull().sum()} rows with missing target values")
        data = data.dropna(subset=[target_column])
    
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Remove target from feature lists
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    # Check for low-cardinality numeric columns (likely categorical)
    for col in numeric_cols.copy():
        if data[col].nunique() < 10:
            print(f"Treating numeric column '{col}' as categorical due to low cardinality")
            categorical_cols.append(col)
            numeric_cols.remove(col)
    
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Step 4: Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # For classification, check class distribution
    if problem_type == 'classification':
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype == 'category':
            print(f"\nEncoding categorical target variable '{target_column}'")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print(f"Classes encoded as: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        class_counts = pd.Series(y).value_counts()
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/len(y):.2%})")
        
        # Check if multiclass or binary
        n_classes = len(class_counts)
        if n_classes > 2:
            print(f"Multi-class classification problem with {n_classes} classes")
        else:
            print("Binary classification problem")
    
    # Step 5: Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Step 6: Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y if problem_type == 'classification' else None
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Step 7: Handle class imbalance for classification
    if problem_type == 'classification':
        class_counts = pd.Series(y_train).value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 1.5:  # If classes are imbalanced
            print(f"\nDetected class imbalance (ratio: {imbalance_ratio:.2f})")
            
            # Auto-select balancing method based on dataset size
            if balance_method == 'auto':
                if len(X_train) < 1000:
                    balance_method = 'smote'
                else:
                    balance_method = 'undersample'
                print(f"Auto-selected balancing method: {balance_method}")
            
            # Apply the selected method
            if balance_method == 'smote':
                print("Applying SMOTE to oversample minority class")
                smote = SMOTE(random_state=random_state)
                # Fit the preprocessor first
                X_train_processed = preprocessor.fit_transform(X_train)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
                # We'll need to handle this differently in the model fitting
                resampled = True
            elif balance_method == 'undersample':
                print("Applying RandomUnderSampler to undersample majority class")
                rus = RandomUnderSampler(random_state=random_state)
                # Fit the preprocessor first
                X_train_processed = preprocessor.fit_transform(X_train)
                X_train_resampled, y_train_resampled = rus.fit_resample(X_train_processed, y_train)
                # We'll need to handle this differently in the model fitting
                resampled = True
            else:
                print("No balancing applied")
                resampled = False
        else:
            print("\nClasses are relatively balanced - no resampling needed")
            resampled = False
    else:
        resampled = False
    
    # Step 8: Define models based on problem type
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'SVM': SVC(probability=True, random_state=random_state),
            'KNN': KNeighborsClassifier(),
            'XGBoost': xgb.XGBClassifier(random_state=random_state),
            'LightGBM': lgb.LGBMClassifier(random_state=random_state)
        }
        
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, -1]
            }
        }
    else:  # regression
        # Define regression models (not implemented in this version)
        print("Regression not implemented in this version")
        return None
    
    # Step 9: Train and evaluate models
    print("\nTraining and evaluating models...")
    
    # Dictionary to store results
    results = {
        'models': {},
        'best_model': None,
        'best_score': 0,
        'metrics': {}
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create full pipeline with preprocessing
        if resampled:
            # For resampled data, we've already applied the preprocessor
            # So we just need to fit the model
            grid_search = GridSearchCV(
                model, param_grids[name], cv=5, 
                scoring='f1_weighted' if problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X_train_resampled, y_train_resampled)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            
            # For prediction, we need to apply preprocessing to test data
            X_test_processed = preprocessor.transform(X_test)
            y_pred = best_model.predict(X_test_processed)
        else:
            # For non-resampled data, use the full pipeline
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Perform grid search
            param_grid = {'model__' + key: value for key, value in param_grids[name].items()}
            grid_search = GridSearchCV(
                full_pipeline, param_grid, cv=5, 
                scoring='f1_weighted' if problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
        
        # Calculate metrics for classification
        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            results['models'][name] = best_model
            results['metrics'][name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'best_params': grid_search.best_params_
            }
            
            # Track best model
            if f1 > results['best_score']:
                results['best_score'] = f1
                results['best_model'] = name
            
            # Print metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Regression metrics (not implemented)
            pass
    
    # Step 10: Compare models
    print("\nModel Comparison:")
    if problem_type == 'classification':
        metrics_df = pd.DataFrame({
            model_name: {
                'Accuracy': results['metrics'][model_name]['accuracy'],
                'Precision': results['metrics'][model_name]['precision'],
                'Recall': results['metrics'][model_name]['recall'],
                'F1 Score': results['metrics'][model_name]['f1_score']
            }
            for model_name in models.keys()
        })
        
        print(metrics_df.T.sort_values('F1 Score', ascending=False))
    else:
        # Regression comparison (not implemented)
        pass
    
    # Step 11: Save the best model
    print(f"\nBest model: {results['best_model']}")
    print(f"Best F1 score: {results['best_score']:.4f}")
    
    # Get the best model
    if resampled:
        # For resampled data, we need to create a pipeline with the preprocessor and best model
        best_model_instance = results['models'][results['best_model']]
        best_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', best_model_instance)
        ])
        
        # Fit the pipeline on the original data
        best_pipeline.fit(X_train, y_train)
        final_model = best_pipeline
    else:
        # For non-resampled data, the best model is already a pipeline
        final_model = results['models'][results['best_model']]
    
    # Save model and metadata
    model_data = {
        'model': final_model,
        'feature_names': X.columns.tolist(),
        'target_column': target_column,
        'best_model_name': results['best_model'],
        'metrics': results['metrics'][results['best_model']],
        'problem_type': problem_type
    }
    
    joblib.dump(model_data, save_model_path)
    print(f"Best model saved to {save_model_path}")
    
    # Create a prediction function
    def predict_new_data(new_data, model_path=save_model_path):
        """
        Function to make predictions on new data using the saved model
        
        Parameters:
        -----------
        new_data : pandas.DataFrame or str
            New data for prediction or path to an Excel file containing new data
        model_path : str
            Path to the saved model file
            
        Returns:
        --------
        numpy.ndarray
            Predicted values and probabilities (if classification)
        """
        # Load model data
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data['feature_names']
        problem_type = model_data.get('problem_type', 'classification')
        
        # Handle if new_data is a file path
        if isinstance(new_data, str):
            if new_data.endswith(('.xls', '.xlsx')):
                new_data = pd.read_excel(new_data)
            else:
                raise ValueError("Unsupported file format. Please use Excel files (.xls, .xlsx).")
        
        # Ensure we have all required features
        missing_cols = [col for col in feature_names if col not in new_data.columns]
        if missing_cols:
            print(f"Warning: Missing columns in new data: {missing_cols}")
            for col in missing_cols:
                new_data[col] = 0  # Add missing columns with default values
        
        # Select only the features used during training
        X_new = new_data[feature_names]
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # For classification, also return probabilities
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new)
            return predictions, probabilities
        else:
            return predictions
    
    # Add prediction function to results
    results['predict_function'] = predict_new_data
    
    return results


# Example usage
if __name__ == "__main__":
    print("Excel ML Processor ready to use!")
    print("Example usage:")
    print("  results = excel_ml_processor('your_data.xlsx', target_column='diagnosis')")
    print("  # Make predictions")
    print("  new_data = pd.read_excel('new_data.xlsx')")
    print("  predictions = results['predict_function'](new_data)")
