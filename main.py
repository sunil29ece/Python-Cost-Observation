import pandas as pd
import matplotlib.pyplot as plt

# Sample monthly cost observations (CAD)
data = {
    "Category": ["Food", "Transport", "Rent", "Utilities", "Shopping"],
    "Cost": [500, 200, 1500, 300, 400],
}

df = pd.DataFrame(data)
total = df["Cost"].sum()
average = df["Cost"].mean()
highest = df.loc[df["Cost"].idxmax()]
lowest = df.loc[df["Cost"].idxmin()]

print("Monthly Cost Observations")
print(df.to_string(index=False))
print("\nCost Analysis")
print(f"Total Cost: ${total:.2f}")
print(f"Average Cost: ${average:.2f}")
print(f"Highest: {highest['Category']} (${highest['Cost']:.2f})")
print(f"Lowest: {lowest['Category']} (${lowest['Cost']:.2f})")

plt.figure(figsize=(9, 5))
plt.bar(df["Category"], df["Cost"])
plt.title("Monthly Cost Observation")
plt.xlabel("Expense Category")
plt.ylabel("Cost (CAD)")
plt.tight_layout()
plt.savefig("cost_analysis.png")
plt.show()
