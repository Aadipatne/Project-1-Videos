import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Objective: Track views, likes, age, and category of videos

# Loading data into a DataFrame
data = pd.read_csv("data.csv")

# Show first 5 rows
print("FIRST 5 ROWS")
print(data.head())

# Getting rows and columns
print("\nROWS AND COLUMNS")
print("No. of Rows:", data.shape[0], "No. of Columns:", data.shape[1])

# Dataset information
print("\nDATA INFO")
data.info()

# Checking for NULL values
print("\nNULL VALUES")
print(data.isna().any())

# Summary statistics
print("\nSTATISTICS")
print(data.describe())

# Data types
print("\nDATA TYPES")
print(data.dtypes)

# Checking duplicates
print("\nDUPLICATES")
print(data.duplicated().sum())

# Category counts
print("\nCATEGORY COUNTS")
print(data["Category"].value_counts())

# Create new column: Engagement Rate
data["EngagementRate"] = data["Likes"] / data["Views"]

print("\nENGAGEMENT RATE")
print(data[["Name", "EngagementRate"]])

# Scatterplot: Views vs Likes
sns.scatterplot(x=data["Views"], y=data["Likes"])
plt.title("Views vs Likes")
plt.xlabel("Views")
plt.ylabel("Likes")
plt.show()

# Scatterplot: Age vs Views
sns.scatterplot(x=data["Age"], y=data["Views"])
plt.title("Age vs Views")
plt.xlabel("Age")
plt.ylabel("Views")
plt.show()

# Pie chart for categories
data["Category"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Video Categories")
plt.ylabel("")
plt.show()

# Bar chart for categories
data["Category"].value_counts().plot(kind="bar")
plt.title("Number of Videos per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Top videos by views
print("\nTOP VIDEOS BY VIEWS")
print(data.sort_values(by=["Views"], ascending=False).head(10)[["Name", "Views"]])

# Top videos by likes
print("\nTOP VIDEOS BY LIKES")
print(data.sort_values(by=["Likes"], ascending=False).head(10)[["Name", "Likes"]])

# Top videos by engagement rate
print("\nTOP VIDEOS BY ENGAGEMENT RATE")
print(data.sort_values(by=["EngagementRate"], ascending=False).head(10)[["Name", "EngagementRate"]])

# Average views by category
print("\nAVERAGE VIEWS BY CATEGORY")
print(data.groupby("Category")["Views"].mean())

# Average likes by category
print("\nAVERAGE LIKES BY CATEGORY")
print(data.groupby("Category")["Likes"].mean())

# Top videos by views visualization
new_index = data.sort_values(by=["Views"], ascending=False).head(10).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_data["Name"], y=sorted_data["Views"])
plt.title("Top Videos by Views")
plt.xlabel("Video Name")
plt.ylabel("Views")
plt.show()

# Top videos by likes visualization
new_index = data.sort_values(by=["Likes"], ascending=False).head(10).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_data["Name"], y=sorted_data["Likes"])
plt.title("Top Videos by Likes")
plt.xlabel("Video Name")
plt.ylabel("Likes")
plt.show()

# Average views by category visualization
avg_views = data.groupby("Category")["Views"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_views["Category"], y=avg_views["Views"])
plt.title("Average Views by Category")
plt.xlabel("Category")
plt.ylabel("Average Views")
plt.show()

# Pairplot
sns.pairplot(data)
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Final conclusions
print("\nEDA CONCLUSIONS")
print("1. The dataset contains video information such as age, views, likes, and category.")
print("2. The most common category is:", data["Category"].value_counts().idxmax())
print("3. The video with the highest views is:", data.loc[data["Views"].idxmax(), "Name"])
print("4. The video with the highest likes is:", data.loc[data["Likes"].idxmax(), "Name"])
print("5. Views and likes have a positive relationship based on the scatterplot and correlation heatmap.")