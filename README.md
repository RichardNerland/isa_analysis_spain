# ISA Analysis Dashboard

A Dash application for analyzing Income Share Agreement (ISA) models for educational programs in Ecuador and Guatemala.

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn isa_dash:server -c gunicorn_config.py`
   - Python Version: 3.9.0

## Local Development

To run the application locally:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python isa_dash.py
```

The application will be available at `http://localhost:8050` 