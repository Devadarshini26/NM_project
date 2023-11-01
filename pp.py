import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load data
data = pd.read_csv("demand.csv")

# Check for missing values
print("Contains any null values or not:")
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Create a scatter plot
fig = px.scatter(data, x="Units Sold", y="Total Price", size='Units Sold', color_discrete_sequence=['red'])
fig.update_traces(marker=dict(color='green'))
fig.show()

# Calculate and display correlation
print("\nCorrelation between the features of the dataset:")
correlations = data.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

# Prepare data for modeling
X = data[["Total Price", "Base Price"]]
X.columns = ["Total Price", "Base Price"]  # Set feature names
y = data["Units Sold"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Regressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\nR-squared scores for Train and Test:")
print(f"Training R-squared score: {train_score}")
print(f"Testing R-squared score: {test_score}")

# Make a prediction
features = np.array([[133.00, 140.00]])
prediction = model.predict(features)


print(f"\nPredicted Units Sold for Total Price={"Total Price"} and Base Price={"Base Price"}: {prediction[0]}")
