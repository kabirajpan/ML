import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Creating the dataset
data = pd.DataFrame({
    "Area": [500, 700, 1200, 1500, 1800],  # Feature (X)
    "Price": [200000, 250000, 400000, 500000, 600000]  # Target (y)
})

# Splitting into features (X) and target (y)
X = data[["Area"]]  # Features must be 2D
y = data["Price"]    # Target (1D)

# Training the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions for visualization
X_range = [[min(X["Area"])], [max(X["Area"])]]  # Min & Max area values
y_pred = model.predict(X_range)  # Predict prices for these values

# Plot the data points
plt.scatter(X, y, color="blue", label="Actual Data")  # Scatter plot of actual data

# Plot the regression line
plt.plot(X_range, y_pred, color="red", linestyle="--", label="Regression Line")

# Making a prediction for a 1600 sq ft house
predicted_price = model.predict([[1600]])
print(f"Predicted price for 1600 sq ft: ${predicted_price[0]:,.2f}")

# Plot predicted point
plt.scatter(1600, predicted_price, color="green", marker="*", s=200, label="Predicted (1600 sq ft)")

# Labels & Title
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression: Area vs Price")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
