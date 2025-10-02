# explore_data.py
import pandas as pd
import urllib.request

# Download the data
print("Downloading dataset...")
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
filename = "car_fuel_efficiency.csv"
urllib.request.urlretrieve(url, filename)

# Read the data
df = pd.read_csv(filename)

print("Dataset columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nDataset info:")
print(df.info())