Title: Precision Market Analytics: Customer Segmentation and Localized Sales Forecasting

Problem Statement:E-commerce and retail companies generate massive amounts of customer data. They struggle to group users based on purchasing behavior and subsequently predict how much specific demographic segments will spend.

Objective:To build a two-phase hybrid pipeline. Phase 1 clusters customers into distinct behavioral groups to compare algorithms. Phase 2 isolates a specific cluster and applies non-parametric regression to predict future spending based on localized trends.

Algorithms Used:
1. K-Means Clustering: To group customers based on hard boundaries.
2. Expectation-Maximization (EM / Gaussian Mixture Model): To group customers based on soft probabilities.
3. Locally Weighted Regression (LWR): To fit a flexible, localized curve predicting a customer's spending score based on their income.

Dataset Suggestion: The Mall_Customers.csv dataset (Features: Age, Annual Income, Spending Score).

Complete Python Code:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# --- PHASE 1: CLUSTERING ---

# 1. Create a simulated Customer Dataset (if you don't have Mall_Customers.csv)
np.random.seed(42)
n_samples = 200
income = np.random.uniform(15, 140, n_samples)
# Create a non-linear relationship for spending score
spending = 50 + 30 * np.sin(income / 10) + np.random.normal(0, 10, n_samples) 
data = pd.DataFrame({'Income': income, 'Spending': spending})

X_cluster = data[['Income', 'Spending']].values

# 2. Apply EM (Gaussian Mixture) vs K-Means
em = GaussianMixture(n_components=3, random_state=42)
em_labels = em.fit_predict(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster)

print("Phase 1: Clustering Complete. Check visualization for comparisons.")

# --- PHASE 2: LOCALLY WEIGHTED REGRESSION (LWR) ---

# 3. LWR Function (From Lab Manual)
def locally_weighted_regression(x_query, X_train, Y_train, tau):
    X_b = np.c_[np.ones(len(X_train)), X_train] # Add bias term
    x_q_b = np.array([1, x_query])
    m = len(X_train)
    W = np.zeros((m, m))
    
    for i in range(m):
        diff = x_query - X_train[i]
        W[i, i] = np.exp(-(diff**2) / (2 * tau**2))
        
    # theta = (X^T * W * X)^-1 * X^T * W * Y
    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ Y_train
    return np.dot(x_q_b, theta)

# 4. Apply LWR to predict Spending based on Income
X_reg = data['Income'].values
Y_reg = data['Spending'].values
tau = 5.0 # Bandwidth

# Generate predictions for a smooth line
X_test = np.linspace(min(X_reg), max(X_reg), 100)
Y_pred = [locally_weighted_regression(x, X_reg, Y_reg, tau) for x in X_test]

# 5. Visualizations
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot Clustering (K-Means)
axs[0].scatter(X_cluster[:, 0], X_cluster[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
axs[0].set_title("Customer Segments (K-Means)")
axs[0].set_xlabel("Annual Income")
axs[0].set_ylabel("Spending Score")

# Plot Regression (LWR)
axs[1].scatter(X_reg, Y_reg, color='gray', alpha=0.5, label='Actual Customers')
axs[1].plot(X_test, Y_pred, color='red', linewidth=2, label='LWR Trend Line')
axs[1].set_title("Localized Sales Forecasting (LWR)")
axs[1].set_xlabel("Annual Income")
axs[1].set_ylabel("Spending Score")
axs[1].legend()

plt.tight_layout()
plt.show()

Step-by-Step Explanation:
1. Clustering: We use Income and Spending data to find distinct buyer personas . We compare standard K-Means with the EM algorithm.
2. LWR Implementation: Because customer spending habits are non-linear (e.g., middle-income might spend less relative to both high and low-income demographics), standard linear regression fails. We use the custom LWR formula  which gives more "weight" to data points near the query point.
3. Visualization: We plot the clusters, and then separately plot the LWR curve cutting dynamically through the customer data .

Output Explanation:
1. Left Graph: Shows 3 distinct customer groups. You can target marketing based on these colors.
2. Right Graph: The red line represents the predicted spending score. Notice how the line curves and bends locally to follow the actual data density, which a straight linear regression line cannot do.Future Improvements:Use ID3 Decision Trees  to automatically generate "If-Then" business rules describing each cluster (e.g., "IF Income > 80 AND Age < 30 THEN Cluster 1").

Viva Questions & Answers:
Q: Difference between EM and K-Means?
A: K-Means uses "hard clustering" (a data point belongs to exactly one cluster). EM uses "soft clustering" (assigns a probability of a data point belonging to each cluster).
Q: What are E-Step and M-Step? 
A: E-Step (Expectation) estimates the probability that each point belongs to a cluster. M-Step (Maximization) updates the cluster parameters (mean and variance) based on those probabilities.
Q: Is LWR parametric or non-parametric? What is the role of bandwidth $\tau$? 
A: It is non-parametric; it does not assume a global function shape. The bandwidth ($\tau$) determines how much surrounding points influence the prediction. A small $\tau$ creates a jagged, overfitted line, while a large $\tau$ makes it behave like standard linear regression.
