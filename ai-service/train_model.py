import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    # Load the processed dataset
    df = pd.read_csv('processed_concrete_data.csv')

    # Define features (X) and target (y)
    # We use age_hours and the injected weather/curing data
    X = df[['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 
            'coarse_agg', 'fine_agg', 'age_hours', 'temperature', 
            'humidity', 'curing_method']]
    y = df['strength']

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training CastOpt AI Prediction Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model for use in the API
    joblib.dump(model, 'model.pkl')
    print("✅ Model trained and saved as 'model.pkl'")

if __name__ == "__main__":
    train_model()