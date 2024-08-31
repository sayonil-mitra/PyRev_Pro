from flask import Blueprint, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

debt_to_equity_ratio_bp = Blueprint('debt_to_equity_ratio', __name__)

@debt_to_equity_ratio_bp.route('/debt-to-equity-ratio', methods=['POST'])
def process_debt_to_equity_ratio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Load CSV file into DataFrame
        data = pd.read_csv(file)

        # Step 1: Generate realistic data
        def generate_realistic_data(num_years=10):
            initial_debt = 100000
            initial_equity = 150000

            np.random.seed(42)  # For reproducibility

            years = np.arange(2020, 2020 + num_years)
            debt_growth_rate = np.random.uniform(0.02, 0.1, num_years)
            equity_growth_rate = np.random.uniform(0.02, 0.1, num_years)
            interest_rate = np.random.uniform(0.03, 0.07, num_years)
            revenue_growth = np.random.uniform(0.03, 0.08, num_years)
            operating_income = np.random.uniform(15000, 30000, num_years)
            asset_growth = np.random.uniform(0.02, 0.06, num_years)
            liabilities_growth = np.random.uniform(0.01, 0.05, num_years)

            debt = [initial_debt]
            equity = [initial_equity]

            for i in range(1, num_years):
                debt.append(debt[-1] * (1 + debt_growth_rate[i-1]))
                equity.append(equity[-1] * (1 + equity_growth_rate[i-1]))

            data = pd.DataFrame({
                'year': years,
                'debt': debt,
                'equity': equity,
                'interest_rate': interest_rate,
                'revenue_growth': revenue_growth,
                'operating_income': operating_income,
                'asset_growth': asset_growth,
                'liabilities_growth': liabilities_growth
            })

            return data

        data = generate_realistic_data()

        # Step 3: Calculate growth rates and volatilities for additional factors
        data['debt_growth'] = data['debt'].pct_change()
        data['equity_growth'] = data['equity'].pct_change()
        data['interest_rate_growth'] = data['interest_rate'].pct_change()
        data['revenue_growth_change'] = data['revenue_growth'].pct_change()
        data['operating_income_growth'] = data['operating_income'].pct_change()
        data['asset_growth_change'] = data['asset_growth'].pct_change()
        data['liabilities_growth_change'] = data['liabilities_growth'].pct_change()

        annual_debt_growth_rate = data['debt_growth'].mean()
        annual_equity_growth_rate = data['equity_growth'].mean()
        debt_growth_volatility = data['debt_growth'].std()
        equity_growth_volatility = data['equity_growth'].std()

        def monte_carlo_debt_to_equity(num_simulations, num_years, historical_debt, historical_equity,
                                       annual_debt_growth_rate, annual_equity_growth_rate,
                                       debt_growth_volatility, equity_growth_volatility,
                                       distribution='normal'):
            debt_to_equity_ratios = []

            for _ in range(num_simulations):
                debt = historical_debt
                equity = historical_equity
                debt_ratios = []

                for year in range(num_years):
                    if distribution == 'normal':
                        debt_growth = np.random.normal(annual_debt_growth_rate, debt_growth_volatility)
                        equity_growth = np.random.normal(annual_equity_growth_rate, equity_growth_volatility)
                    elif distribution == 'log-normal':
                        debt_growth = np.random.lognormal(mean=np.log(1 + annual_debt_growth_rate),
                                                          sigma=debt_growth_volatility)
                        equity_growth = np.random.lognormal(mean=np.log(1 + annual_equity_growth_rate),
                                                            sigma=equity_growth_volatility)
                    elif distribution == 'uniform':
                        debt_growth = np.random.uniform(annual_debt_growth_rate - debt_growth_volatility,
                                                        annual_debt_growth_rate + debt_growth_volatility)
                        equity_growth = np.random.uniform(annual_equity_growth_rate - equity_growth_volatility,
                                                          annual_equity_growth_rate + equity_growth_volatility)
                    elif distribution == 'triangular':
                        debt_growth = np.random.triangular(left=annual_debt_growth_rate - debt_growth_volatility,
                                                           mode=annual_debt_growth_rate,
                                                           right=annual_debt_growth_rate + debt_growth_volatility)
                        equity_growth = np.random.triangular(left=annual_equity_growth_rate - equity_growth_volatility,
                                                             mode=annual_equity_growth_rate,
                                                             right=annual_equity_growth_rate + equity_growth_volatility)

                    debt *= (1 + debt_growth)
                    equity *= (1 + equity_growth)
                    debt_ratios.append(debt / equity)

                debt_to_equity_ratios.append(debt_ratios)

            return np.array(debt_to_equity_ratios)

        num_simulations = 100
        num_years = 10
        historical_debt = data['debt'].iloc[-1]
        historical_equity = data['equity'].iloc[-1]

        debt_to_equity_results = monte_carlo_debt_to_equity(num_simulations, num_years, historical_debt,
                                                           historical_equity, annual_debt_growth_rate,
                                                           annual_equity_growth_rate, debt_growth_volatility,
                                                           equity_growth_volatility)

        # Plot both charts in one image
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Histogram of final debt-to-equity ratios
        axs[0].hist(debt_to_equity_results[:, -1], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axs[0].set_title('Monte Carlo Simulation: Debt-to-Equity Ratio Distribution (Final Year)')
        axs[0].set_xlabel('Debt-to-Equity Ratio')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(True)

        # Evolution of debt-to-equity ratio over time for a few simulations
        for i in range(min(num_simulations, 10)):  # Limit number of lines to 10 for clarity
            axs[1].plot(range(1, num_years + 1), debt_to_equity_results[i], color='blue', alpha=0.1)
        axs[1].set_title('Debt-to-Equity Ratio Evolution Over Time (Sample Simulations)')
        axs[1].set_xlabel('Year')
        axs[1].set_ylabel('Debt-to-Equity Ratio')
        axs[1].grid(True)

        # Save plot to a BytesIO object
        img = BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Send the image file in the response
        return send_file(img, mimetype='image/png')
