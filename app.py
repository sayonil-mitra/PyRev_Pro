from flask import Flask
from endpoints.cac_projection import cac_projection_bp
from endpoints.clv_projection import clv_projection_bp
from endpoints.debt_to_equity_ratio import debt_to_equity_ratio_bp
from endpoints.cost_of_ownership import cost_of_ownership_bp

app = Flask(__name__)

# Register the blueprint for the /process-csv endpoint
app.register_blueprint(cac_projection_bp)
app.register_blueprint(clv_projection_bp)
app.register_blueprint(debt_to_equity_ratio_bp)
app.register_blueprint(cost_of_ownership_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
