# 🛒 Precision Market Analytics: Segmentation & Forecasting

## 📌 Project Overview
This project is a two-phase machine learning pipeline designed for retail and e-commerce analytics. It combines unsupervised clustering to discover hidden customer segments with non-parametric regression to forecast dynamic spending behavior. 

## 🚀 Key Features
- **Phase 1 - Customer Segmentation:** Implements and compares K-Means and Expectation-Maximization (Gaussian Mixture) to group customers based on purchasing power.
- **Phase 2 - Sales Forecasting:** Uses Locally Weighted Regression (LWR) to predict customer spending scores based on localized income brackets, handling non-linear retail trends efficiently.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
- **Algorithms:** K-Means, EM (GMM), Locally Weighted Regression (Custom Implementation)

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/Market-Analytics-ML.git](https://github.com/yourusername/Market-Analytics-ML.git)
   ```
2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib
3. Run the analysis script:
   python market_analysis.py

📊 Results & Visualizations
Clustering: Successfully identified 3 distinct buyer personas (e.g., Budget-conscious, High-Rollers, Average Consumers).

Regression: The LWR model (with bandwidth τ = 5.0) successfully mapped the non-linear relationship between annual income and spending score, outperforming standard linear regression.

4. UI Idea: Simple HTML Frontend
For a business analytics tool, the UI should look like a corporate dashboard. This simple Bootstrap HTML provides two main functions: profiling a customer to see which cluster they belong to, and predicting their exact spend.

**`index.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Analytics Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 40px; }
        .card { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: none; margin-bottom: 20px;}
        .header-title { color: #2c3e50; font-weight: bold; }
        .btn-custom { background-color: #007bff; color: white; }
        .btn-custom:hover { background-color: #0056b3; color: white; }
    </style>
</head>
<body>

<div class="container">
    <h2 class="text-center header-title mb-5">📊 Precision Market Analytics Engine</h2>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card p-4 h-100">
                <h4 class="mb-3">🎯 Customer Segmentation (K-Means/EM)</h4>
                <p class="text-muted small">Input customer details to determine their marketing persona segment.</p>
                <form action="/cluster" method="POST">
                    <div class="mb-3">
                        <label class="form-label">Customer Age</label>
                        <input type="number" class="form-control" name="age" placeholder="e.g. 34" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Annual Income ($k)</label>
                        <input type="number" class="form-control" name="income" placeholder="e.g. 75" required>
                    </div>
                    <button type="submit" class="btn btn-custom w-100 mt-2">Assign to Cluster</button>
                </form>
            </div>
        </div>

        <div class="col-md-6">
            <div class="card p-4 h-100">
                <h4 class="mb-3">📈 Spend Forecasting (LWR)</h4>
                <p class="text-muted small">Predict a specific customer's spending score based on localized trends.</p>
                <form action="/predict_spend" method="POST">
                    <div class="mb-3">
                        <label class="form-label">Annual Income ($k)</label>
                        <input type="number" class="form-control" name="target_income" placeholder="e.g. 75" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Algorithm Bandwidth (Tau)</label>
                        <input type="number" step="0.1" class="form-control" name="tau" value="5.0" required>
                        <div class="form-text">Lower tau = more fitted to local data.</div>
                    </div>
                    <button type="submit" class="btn btn-success w-100 mt-2">Predict Spending Score</button>
                </form>
            </div>
        </div>
    </div>
</div>

</body>
</html>
