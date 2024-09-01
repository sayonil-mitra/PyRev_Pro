from flask import Blueprint, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

cost_of_ownership_bp = Blueprint('cost_of_ownership', __name__)

@cost_of_ownership_bp.route('/cost-of-ownership', methods=['POST'])
def process_cost_of_ownership():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Load CSV file into DataFrame
        data = pd.read_csv(file)

        # Step 1: Extract data for simulation
        initial_cost = data['cost'].values[0]
        annual_growth_rate = data['cost_growth_rate'].mean()
        volatility = data['cost_volatility'].mean()
        num_simulations = 1000
        num_years = 10
        distribution = 'normal'  # Change to 'log-normal', 'uniform', or 'triangular' as needed

        # Monte Carlo simulation function for TCO
        def monte_carlo_tco(num_simulations, num_years, initial_cost, annual_growth_rate, volatility, distribution='normal'):
            tco_values = []

            for _ in range(num_simulations):
                cost = initial_cost
                yearly_costs = []

                for year in range(num_years):
                    if distribution == 'normal':
                        growth = np.random.normal(loc=annual_growth_rate, scale=volatility)
                    elif distribution == 'log-normal':
                        growth = np.random.lognormal(mean=np.log(1 + annual_growth_rate), sigma=volatility)
                    elif distribution == 'uniform':
                        growth = np.random.uniform(low=annual_growth_rate - volatility, high=annual_growth_rate + volatility)
                    elif distribution == 'triangular':
                        growth = np.random.triangular(left=annual_growth_rate - volatility, mode=annual_growth_rate, right=annual_growth_rate + volatility)

                    cost *= (1 + growth)
                    yearly_costs.append(cost)

                tco_values.append(yearly_costs)

            return np.array(tco_values)

        # Run the simulation
        results = monte_carlo_tco(num_simulations, num_years, initial_cost, annual_growth_rate, volatility, distribution)

        # Calculate statistics
        final_year_costs = results[:, -1]
        mean_costs = np.mean(results, axis=0)
        median_costs = np.median(results, axis=0)
        std_deviation = np.std(results, axis=0)

        # Plot both charts in one image
        fig, axs = plt.subplots(1, 2, figsize=(14, 8))

        # Histogram of total costs for the final year
        axs[0].hist(final_year_costs, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axs[0].set_title('Monte Carlo Simulation: Total Cost of Ownership Distribution (Final Year)')
        axs[0].set_xlabel('Total Cost of Ownership ($)')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(True)

        # Mean and standard deviation over time
        years_range = np.arange(1, num_years + 1)
        axs[1].plot(years_range, mean_costs, label='Mean Cost', color='blue')
        axs[1].fill_between(years_range,
                            mean_costs - std_deviation,
                            mean_costs + std_deviation,
                            color='blue', alpha=0.2, label='±1 Std Dev')
        axs[1].set_title('Total Cost of Ownership Over Time')
        axs[1].set_xlabel('Year')
        axs[1].set_ylabel('Total Cost of Ownership ($)')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()

        # Save plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Send the image file in the response
        return send_file(img, mimetype='image/png')
