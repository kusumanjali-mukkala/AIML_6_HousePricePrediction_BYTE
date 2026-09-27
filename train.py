import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("data/house_prices.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# -----------------------------
# 2. Separate input and target
# -----------------------------
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: MedHouseVal")

# -----------------------------
# 3. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 4. Create Linear Regression model
# -----------------------------
model = LinearRegression()

# -----------------------------
# 5. Train model
# -----------------------------
model.fit(X_train, y_train)

print("\nModel training completed!")

# -----------------------------
# 6. Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Evaluate model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n----- Model Performance -----")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# -----------------------------
# 8. Save model
# -----------------------------
joblib.dump(model, "model/linear_regression_model.pkl")

print("\nModel saved successfully!")
print("Location: model/linear_regression_model.pkl")

# -----------------------------
# 9. Save predictions
# -----------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results.to_csv("outputs/predictions.csv", index=False)

# -----------------------------
# 10. Save training results
# -----------------------------
with open("outputs/training_results.txt", "w") as file:
    file.write("Linear Regression House Price Prediction\n")
    file.write("========================================\n\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n\n")
    file.write(f"MAE: {mae:.4f}\n")
    file.write(f"MSE: {mse:.4f}\n")
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"R2 Score: {r2:.4f}\n")

print("Training results saved!")