import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import numpy as np

# Suppress warnings globally
warnings.filterwarnings('ignore')

def main():
    # Load balanced dataset - using the current path
    data = pd.read_csv('C:/Users/hp/Desktop/App_yassir/heart_failure_balanced.csv')
    X = data.drop('DEATH_EVENT', axis=1)
    y = data['DEATH_EVENT']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("\nRandom Forest model trained and evaluated!")
    print(f"Test Set Accuracy: {test_accuracy:.3f}")

    # Print feature importance
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df)

    # Save the model and scaler
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("\nRandom Forest model and scaler saved!")

if __name__ == "__main__":
    main()
