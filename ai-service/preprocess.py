import pandas as pd
import numpy as np

def preprocess_data():
    print("Reading raw concrete data...")
    # Load the file you just generated
    df = pd.read_csv('concrete_data.csv')

    # 1. Rename columns to shorter, coder-friendly names
    df.columns = [
        'cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 
        'coarse_agg', 'fine_agg', 'age_days', 'strength'
    ]

    # 2. Convert Age from Days to Hours (L&T cycle times are usually in hours)
    df['age_hours'] = df['age_days'] * 24

    # 3. Inject Synthetic Weather & Curing Data
    # This simulates different site conditions (Delhi winter vs Mumbai summer)
    np.random.seed(42)
    num_rows = len(df)

    # Ambient Temperature (5°C to 45°C)
    df['temperature'] = np.random.uniform(5, 45, size=num_rows)

    # Humidity (20% to 90%)
    df['humidity'] = np.random.uniform(20, 90, size=num_rows)

    # Curing Method (0: Natural Air, 1: Steam Curing, 2: Chemical Membrane)
    # We weight this so Natural Air (0) is the most common baseline
    df['curing_method'] = np.random.choice([0, 1, 2], size=num_rows, p=[0.7, 0.2, 0.1])

    # 4. Save the new "L&T Ready" dataset
    output_file = 'processed_concrete_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Success: '{output_file}' created!")
    print(f"New Columns: {list(df.columns)}")

if __name__ == "__main__":
    preprocess_data()