from flask import Blueprint, request, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import io

# Define a blueprint for this endpoint
cac_projection_bp = Blueprint('cac_projection', __name__)  # Updated blueprint name

# Function to generate realistic data
def generate_realistic_data():
    np.random.seed(42)  # For reproducibility
    num_years = 10
    initial_marketing_expense = 500000
    initial_customers = 1000000
    initial_cac = 200000
    years = np.arange(2014, 2014 + num_years)

    # Generate random variations
    marketing_expenses = initial_marketing_expense * (1 + np.random.uniform(0.05, 0.2, num_years).cumsum())
    customers = initial_customers * (1 + np.random.uniform(0.01, 0.1, num_years).cumsum())
    cac = initial_cac * (1 + np.random.uniform(-0.05, 0.05, num_years))
    revenue_growth_rate = np.random.uniform(0.02, 0.4, num_years)
    customer_conversion_rate = np.random.uniform(0.01, 0.05, num_years)
    churn_rate = np.random.uniform(0.1, 0.3, num_years)

    data = pd.DataFrame({
        'Year': years,
        'Sales & Marketing Expenses': marketing_expenses,
        'Customer Acquisition Cost': cac,
        'Revenue Growth Rate': revenue_growth_rate,
        'Customer Conversion Rate': customer_conversion_rate,
        'Total Customers': customers,
        'Customer Churn Rate': churn_rate
    })

    return data

# Function to load or generate data
def load_or_generate_data(csv_file=None):
    if csv_file:
        data = pd.read_csv(csv_file)
    else:
        data = generate_realistic_data()
    return data

@cac_projection_bp.route('/cac', methods=['POST'])
def process_csv():
    file = request.files.get('file')
    
    # Load or generate data based on the uploaded file
    data = load_or_generate_data(file)

    # Step 3: Feature Engineering
    data['Marketing Efficiency Ratio'] = data['Sales & Marketing Expenses'] / data['Total Customers']
    data['CAC Change Rate'] = data['Customer Acquisition Cost'].pct_change().fillna(0)
    data['Marketing Spend Change Rate'] = data['Sales & Marketing Expenses'].pct_change().fillna(0)

    # Step 4: Prepare Data for Modeling
    X = data[['Sales & Marketing Expenses', 'Total Customers', 'Revenue Growth Rate', 'Customer Conversion Rate',
              'Customer Churn Rate', 'Marketing Efficiency Ratio', 'CAC Change Rate', 'Marketing Spend Change Rate']]
    y = data['Customer Acquisition Cost']

    # Step 5: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Model Training (Gradient Boosting)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Step 7: Model Evaluation
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a plot to visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data['Customer Acquisition Cost'], label='Actual CAC', marker='o')
    plt.plot(data['Year'][len(X_train):], y_pred, label='Predicted CAC', marker='x')
    plt.title('Actual vs. Predicted Customer Acquisition Cost')
    plt.xlabel('Year')
    plt.ylabel('Customer Acquisition Cost')
    plt.legend()

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Return the image file as a response
    return send_file(buf, mimetype='image/png')
