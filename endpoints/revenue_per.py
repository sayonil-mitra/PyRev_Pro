from flask import Blueprint, request, send_file, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO

revenue_per = Blueprint('revenue_projection', __name__)

@revenue_per.route('/revenue-per', methods=['POST'])
def process_revenue_forecast():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Load CSV file into DataFrame
        data = pd.read_csv(file)

        # Data Preprocessing
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
        data.set_index('TransactionDate', inplace=True)
        data.sort_index(inplace=True)

        # Aggregate data by month
        monthly_data = data.resample('M').sum()

        # Fit ARIMA Model
        model = ARIMA(monthly_data['TransactionAmount'], order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast future revenue
        forecast_steps = 12  # Forecast for the next 12 months
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

        # Create DataFrame for forecast
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecast': forecast
        })
        forecast_df.set_index('Date', inplace=True)

        # Visualize Results
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_data.index, monthly_data['TransactionAmount'], label='Historical Data', color='blue')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
        plt.title('Revenue Forecast')
        plt.xlabel('Date')
        plt.ylabel('Transaction Amount')
        plt.legend()

        # Save plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Send the image file in the response
        return send_file(img, mimetype='image/png')
