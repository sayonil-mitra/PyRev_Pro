from flask import Flask
from flask_cors import CORS  # Import the CORS class
from endpoints.cac_projection import cac_projection_bp
from endpoints.clv_projection import clv_projection_bp
from endpoints.debt_to_equity_ratio import debt_to_equity_ratio_bp
from endpoints.cost_of_ownership import cost_of_ownership_bp
from endpoints.revenue_per import revenue_per
from endpoints.ebitda import ebitda_bp

app = Flask(__name__)

# Enable CORS for the entire Flask app
CORS(app)  # This will allow all domains by default

# Alternatively, to allow specific origins, you can use:
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Register the blueprint for the /process-csv endpoint
app.register_blueprint(cac_projection_bp)
app.register_blueprint(clv_projection_bp)
app.register_blueprint(debt_to_equity_ratio_bp)
app.register_blueprint(cost_of_ownership_bp)
app.register_blueprint(revenue_per)
app.register_blueprint(ebitda_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
