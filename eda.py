import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Show first rows
print("FIRST 5 ROWS")
print(df.head())

# Shape of dataset
print("\nSHAPE")
print(df.shape)

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Summary statistics
print("\nSUMMARY STATS")
print(df.describe())

# Category counts
print("\nCATEGORY COUNTS")
print(df["Category"].value_counts())

# Histogram of Views
plt.hist(df["Views"])
plt.title("Views Distribution")
plt.show()

# Views vs Likes
plt.scatter(df["Views"], df["Likes"])
plt.xlabel("Views")
plt.ylabel("Likes")
plt.title("Views vs Likes")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()