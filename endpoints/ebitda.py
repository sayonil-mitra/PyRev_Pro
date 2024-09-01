from flask import Blueprint, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.impute import SimpleImputer
from io import BytesIO
import requests

ebitda_bp = Blueprint('ebitda', __name__)

# Function to fetch data from World Bank API for specific economic indicators
def fetch_world_bank_data(indicator, years, country_code):
    base_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    economic_data = []

    for year in years:
        params = {
            "format": "json",
            "date": year
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1 and data[1]:
                economic_data.append(data[1][0].get('value', np.nan))
            else:
                economic_data.append(np.nan)
        else:
            economic_data.append(np.nan)
            print(f"Failed to retrieve data for {country_code}: {response.status_code}")

    return economic_data

# Fetch real-time or historical data for Economic Indicators
def get_economic_indicators(country_code, years):
    indicators = {
        'InflationRate': "FP.CPI.TOTL.ZG",     # Inflation Rate
        'GDPGrowthRate': "NY.GDP.MKTP.KD.ZG",  # GDP Growth Rate
        'InterestRate': "FR.INR.RINR",         # Real Interest Rate
        'UnemploymentRate': "SL.UEM.TOTL.ZS"   # Unemployment rate
    }

    economic_data = {}

    for key, indicator in indicators.items():
        economic_data[key] = fetch_world_bank_data(indicator, years, country_code)

    return economic_data

@ebitda_bp.route('/ebitda', methods=['POST'])
def process_ebitda():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Load CSV file into DataFrame
        data = pd.read_csv(file)

        # Generate quarterly data from 2015 to 2024 (10 years)
        years = list(range(2015, 2025))
        quarters = pd.date_range(start='2015-01-01', periods=len(years) * 4, freq='Q')

        # Generate user input financial data with variability
        data_model = {
            "financial_data": {
                "Revenue": (np.random.uniform(500000, 700000, len(quarters)) * (1 + np.cumsum(np.random.normal(0.01, 0.02, len(quarters))))).astype(int),
                "COGS": (np.random.uniform(200000, 400000, len(quarters)) * (1 + np.cumsum(np.random.normal(0.01, 0.02, len(quarters))))).astype(int),
                "OperatingExpenses": np.random.randint(50000, 100000, len(quarters)),
                "DepreciationAmortization": np.random.randint(3000, 8000, len(quarters))
            }
        }

        # Fetch economic indicators for 'USA' (example country) for the years 2015 to 2024
        economic_indicators = get_economic_indicators('USA', years)
        economic_df = pd.DataFrame(economic_indicators, index=pd.date_range(start='2015-01-01', periods=len(years), freq='Y'))

        # Prepare the DataFrame for EBITDA Margin
        df = pd.DataFrame(data_model["financial_data"], index=quarters)

        # Add economic indicators to the DataFrame and resample to match the quarterly data
        df = pd.concat([df, economic_df.resample('Q').ffill().iloc[:len(df)]], axis=1)

        # Handle missing values by imputing with median
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

        # Calculate EBITDA
        df_imputed['EBITDA'] = df_imputed['Revenue'] - df_imputed['COGS'] - df_imputed['OperatingExpenses'] + df_imputed['DepreciationAmortization']

        # Calculate EBITDA Margin
        df_imputed['EBITDAMargin'] = (df_imputed['EBITDA'] / df_imputed['Revenue']) * 100

        # Adjust EBITDA Margin with economic indicators
        df_imputed['AdjustedEBITDAMargin'] = df_imputed['EBITDAMargin'] * (1 + df_imputed['GDPGrowthRate'].fillna(0) / 100)

        # Forecast Future EBITDA Margin
        result = adfuller(df_imputed['AdjustedEBITDAMargin'])
        d = 0 if result[1] <= 0.05 else 1

        m = 4  # Quarterly seasonality
        D = 1 if adfuller(df_imputed['AdjustedEBITDAMargin'].diff(m).dropna())[1] > 0.05 else 0
        P, Q = 1, 1

        model = SARIMAX(df_imputed['AdjustedEBITDAMargin'], order=(1, d, 1), seasonal_order=(P, D, Q, m))
        model_fit = model.fit()

        forecast_steps = 20  # Number of periods to forecast (e.g., next 5 years or 20 quarters)
        forecasted_ebitda_margin = model_fit.forecast(steps=forecast_steps)

        # Plotting the forecasted EBITDA Margin
        plt.figure(figsize=(10, 6))
        plt.plot(df_imputed.index, df_imputed["AdjustedEBITDAMargin"], marker='o', label="Historical Adjusted EBITDA Margin")
        plt.plot(pd.date_range(df_imputed.index[-1], periods=forecast_steps+1, freq='Q')[1:],
                 forecasted_ebitda_margin, marker='x', linestyle='--', color='red', label="Forecasted EBITDA Margin")
        plt.title('EBITDA Margin Forecast Using SARIMA')
        plt.xlabel('Date')
        plt.ylabel('Adjusted EBITDA Margin (%)')
        plt.legend()
        plt.grid(True)

        # Save plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Send the image file in the response
        return send_file(img, mimetype='image/png')

    return jsonify({"error": "Invalid request"}), 400
