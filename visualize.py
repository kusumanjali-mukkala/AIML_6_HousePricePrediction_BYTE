import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
data = pd.read_csv("outputs/predictions.csv")

actual = data["Actual"]
predicted = data["Predicted"]

# --------------------------------
# 1. Actual vs Predicted
# --------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(actual, predicted, alpha=0.5)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Values")

plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
plt.close()

print("Saved: outputs/actual_vs_predicted.png")

# --------------------------------
# 2. Residual Plot
# --------------------------------
residuals = actual - predicted

plt.figure(figsize=(8, 6))

plt.scatter(predicted, residuals, alpha=0.5)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted House Value")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.tight_layout()
plt.savefig("outputs/residual_plot.png", dpi=300)
plt.close()

print("Saved: outputs/residual_plot.png")

print("\nVisualization completed successfully!")