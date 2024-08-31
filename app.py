from flask import Flask
from endpoints.cac_projection import cac_projection_bp  # Updated import

app = Flask(__name__)

# Register the blueprint for the /process-csv endpoint
app.register_blueprint(cac_projection_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
