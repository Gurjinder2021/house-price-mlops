
import pandas as pd

# Load dataset
df = pd.read_csv("data/house_prices.csv")

# Display first five records
print("\nFirst five records:")
print(df.head())

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Column names and data types
print("\nDataset information:")
df.info()

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())
