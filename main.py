import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, poisson, chisquare, skew

# Load dataset
crypto = pd.read_csv('crypto_data.csv')
# Define variables
cols = [
    'Customers', 'Website_Visits', 'Crypto_Low', 'Coin_Age_Years',
    'Crypto_Open', 'Active_Users', 'Crypto_Transactions'
]
customer, website_visits, crypto_low, coin_age_years, crypto_open, active_users, test_scores = crypto[cols].T.values

# Helper Functions
def plot_hist(data, title, xlabel=None, bins=8, color='skyblue'):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel if xlabel else title)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def test_poisson(data, label="Data"):
    values, counts = np.unique(data, return_counts=True)
    lambda_ = np.mean(data)
    expected = len(data) * poisson.pmf(values, lambda_)
    expected *= np.sum(counts) / np.sum(expected)
    chi_stat, p_value = chisquare(f_obs=counts, f_exp=expected)
    print(f"\n🔹 {label} Poisson Test")
    print(f"Chi-square Statistic: {chi_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Likely Poisson Distribution" if p_value > 0.05 else "Not Poisson Distributed")

def test_normality(data, label="Data"):
    stat, p = normaltest(data)
    print(f"\n🔹 {label} Normality Test")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
    print("Data is likely normal" if p > 0.05 else " Data is not normal")


# Exploratory Analysis

# Example Poisson test for Customers
plot_hist(customer, "Customers Distribution", bins=5)
test_poisson(customer, "Customers")

# Example Normality test for Transactions
plot_hist(test_scores, "Crypto Transactions (Test Scores)", bins=8)
test_normality(test_scores, "Crypto Transactions")

# Distribution Plots for All Columns

for col in cols:
    plot_hist(crypto[col], f"{col} Distribution")

# Skewness Example
print(f"\n🔹 Active Users Skewness: {skew(active_users):.4f}")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(crypto.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Crypto Dataset")
plt.tight_layout()
plt.show()
