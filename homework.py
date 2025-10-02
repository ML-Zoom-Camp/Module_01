# homework_corrected.py
import pandas as pd
import numpy as np
import urllib.request

# Download the data
print("Downloading dataset...")
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
filename = "car_fuel_efficiency.csv"
urllib.request.urlretrieve(url, filename)

# Read the data
df = pd.read_csv(filename)

print("Dataset loaded successfully!")
print("\n" + "="*50)

# Q1: Pandas version
print("Q1. Pandas version:", pd.__version__)

# Q2: Records count
print(f"\nQ2. Records count: {len(df)}")

# Q3: Fuel types
fuel_types_count = df['fuel_type'].nunique()
print(f"Q3. Number of fuel types: {fuel_types_count}")

# Q4: Missing values
missing_columns = df.isnull().any()
columns_with_missing = missing_columns.sum()
print(f"Q4. Columns with missing values: {columns_with_missing}")

# Q5: Max fuel efficiency from Asia (using fuel_efficiency_mpg)
asia_max_eff = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
print(f"Q5. Max fuel efficiency from Asia: {asia_max_eff}")

# Q6: Horsepower analysis
print("\nQ6. Horsepower analysis:")
horsepower_median_original = df['horsepower'].median()
horsepower_mode = df['horsepower'].mode()[0]
print(f"  Original median: {horsepower_median_original}")
print(f"  Most frequent value: {horsepower_mode}")

# Fill missing values and recalculate median
df_filled = df.copy()
df_filled['horsepower'].fillna(horsepower_mode, inplace=True)
horsepower_median_new = df_filled['horsepower'].median()
print(f"  New median after filling: {horsepower_median_new}")

if horsepower_median_new > horsepower_median_original:
    print("  Answer: Yes, it increased")
elif horsepower_median_new < horsepower_median_original:
    print("  Answer: Yes, it decreased")
else:
    print("  Answer: No")

# Q7: Linear regression calculation - FIXED VERSION
print("\nQ7. Linear regression calculation:")

# Select Asian cars, specific columns, and first 7 values
asia_cars = df_filled[df_filled['origin'] == 'Asia']
selected = asia_cars[['vehicle_weight', 'model_year']].head(7)
X = selected.values

print("  Selected data:")
print(selected)

# The key fix: We need to use the ORIGINAL data (not scaled) but be careful with the calculation
# Add a column of ones for the intercept term
X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

# Compute XTX = X^T * X
XTX = X_with_intercept.T @ X_with_intercept

# Invert XTX
XTX_inv = np.linalg.inv(XTX)

# Create y array
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Compute w = (X^T * X)^-1 * X^T * y
w = XTX_inv @ X_with_intercept.T @ y

# Sum of all elements of w
w_sum = w.sum()
print(f"  Sum of all elements of w: {w_sum:.3f}")

# The issue might be numerical precision - let's check the actual values
print(f"  w vector: {w}")

print("\n" + "="*50)
print("ANSWERS SUMMARY:")
print(f"Q1: {pd.__version__}")
print(f"Q2: {len(df)}")
print(f"Q3: {fuel_types_count}")
print(f"Q4: {columns_with_missing}")
print(f"Q5: {asia_max_eff}")
print(f"Q6: {'Yes, it increased' if horsepower_median_new > horsepower_median_original else 'Yes, it decreased' if horsepower_median_new < horsepower_median_original else 'No'}")
print(f"Q7: {w_sum:.3f}")