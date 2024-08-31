from flask import Blueprint, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO

clv_projection_bp = Blueprint('clv_projection', __name__)

@clv_projection_bp.route('/clv', methods=['POST'])
def process_clv():
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
        current_date = data['TransactionDate'].max() + pd.DateOffset(1)

        # Create summary statistics for each customer
        summary = data.groupby('CustomerID').agg({
            'TransactionDate': lambda x: (current_date - x.max()).days,
            'TransactionAmount': ['mean', 'count']
        }).reset_index()

        # Flatten multi-level columns
        summary.columns = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']

        # Standardize Data
        scaler = StandardScaler()
        X = scaler.fit_transform(summary[['Recency', 'Frequency', 'MonetaryValue']])

        # Apply K-means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        summary['CustomerSegment'] = kmeans.fit_predict(X)

        # Visualize Results
        plt.figure(figsize=(10, 6))
        plt.scatter(summary['Recency'], summary['Frequency'], c=summary['CustomerSegment'], cmap='viridis', s=100)
        plt.title('Customer Segmentation')
        plt.xlabel('Recency')
        plt.ylabel('Frequency')
        plt.colorbar(label='Customer Segment')

        # Save plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Send the image file in the response
        return send_file(img, mimetype='image/png')
