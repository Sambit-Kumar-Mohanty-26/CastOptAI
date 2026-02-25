import pandas as pd
import requests
import io

def download_dataset():
    # Direct URL to the UCI Concrete Compressive Strength dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    
    print("Fetching dataset from UCI Repository...")
    try:
        response = requests.get(url)
        
        # FIX: Changed status_status to status_code
        if response.status_code == 200:
            # Load Excel into Pandas
            df = pd.read_excel(io.BytesIO(response.content))
            
            # Save as the CSV file our plan expects
            df.to_csv('concrete_data.csv', index=False)
            print("✅ Success: 'concrete_data.csv' is ready in the ai-service folder.")
        else:
            print(f"❌ Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    download_dataset()