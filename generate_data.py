
import pandas as pd
import numpy as np
import os

# Reproducible random data
np.random.seed(42)

# Number of houses
n = 500

# Generate house features
data = {
    "area_sqft": np.random.randint(600, 3001, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 5, n),
    "house_age": np.random.randint(0, 31, n)
}

df = pd.DataFrame(data)

# Generate house prices
df["price"] = (
    800000
    + df["area_sqft"] * 3500
    + df["bedrooms"] * 250000
    + df["bathrooms"] * 300000
    - df["house_age"] * 25000
)

# Add random noise
noise = np.random.normal(0, 500000, n)
df["price"] = df["price"] + noise

# Create data folder if it doesn't exist
os.makedirs("src/data", exist_ok=True)

# Save dataset
df.to_csv("data/house_prices.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Dataset shape:", df.shape)
