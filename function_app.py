import azure.functions as func
import logging
import xgboost as xgb
import numpy as np
import os

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="predict", methods=["GET", "POST"])
def result_page(req: func.HttpRequest) -> func.HttpResponse:
    try:
        feature_names = [
            ("median_income", "Median Income"),
            ("house_age", "House Age"),
            ("avg_rooms", "Average Rooms"),
            ("avg_bedrooms", "Average Bedrooms"),
            ("population", "Population"),
            ("avg_occupancy", "Average Occupancy"),
            ("latitude", "Latitude"),
            ("longitude", "Longitude")
        ]

        features = []
        prediction = None

        if req.method == "POST":
            for key, _ in feature_names:
                value = req.form.get(key)
                features.append(float(value) if value else 0)
            features_array = np.array(features).reshape(1, -1)

            MODEL_PATH = os.path.join(os.getcwd(), "models", "xgb_hy.json")
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(MODEL_PATH)
            prediction = xgb_model.predict(features_array)[0]

        # Build table rows including predicted price
        table_rows = ""
        if features:
            for i, (_, label) in enumerate(feature_names):
                table_rows += f"<tr><td>{label}</td><td>{features[i]}</td></tr>"
            if prediction is not None:
                table_rows += f"""
                <tr class='prediction-row'>
                    <td colspan='2'>Predicted Price: ${prediction:.2f}k</td>
                </tr>"""

        html_content = f"""
        <html>
            <head>
                <title>California House Price Predictor</title>
                <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
                <style>
                    body {{
                        background-color: #1F4060; /* Cello */
                        font-family: 'Roboto', sans-serif;
                        color: #F2E7DE; /* milky white */
                        margin: 0;
                        padding: 0;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                    }}
                    .container {{
                        display: flex;
                        gap: 30px;
                        width: 90%;
                        max-width: 1200px;
                        padding: 20px;
                    }}
                    .card {{
                        background-color: #3B5B8C; /* Chambray */
                        padding: 30px;
                        border-radius: 20px;
                        flex: 1;
                        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
                    }}
                    h1 {{
                        text-align: center;
                        font-size: 3em;
                        background: linear-gradient(90deg, #5C7999, #A7C2D3, #F2E7DE); /* Waikawa Gray -> Casper -> Merino */
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-bottom: 25px;
                    }}
                    h2 {{
                        font-size: 2em;
                        text-align: center;
                        margin-bottom: 20px;
                        color: #A7C2D3; /* Casper */
                    }}
                    form {{
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }}
                    label {{
                        font-size: 1.3em;
                        color: #F2E7DE;
                    }}
                    input {{
                        padding: 12px;
                        border-radius: 10px;
                        border: none;
                        background-color: rgba(0,0,0,0.4);
                        color: #F2E7DE;
                        font-size: 1.2em;
                    }}
                    button {{
                        padding: 15px;
                        border-radius: 10px;
                        border: none;
                        background-color: #F2E7DE; /* Merino */
                        color: #1F4060; /* Cello */
                        font-weight: bold;
                        font-size: 1.5em;
                        cursor: pointer;
                        transition: 0.3s;
                        box-shadow: 0 0 10px rgba(242,231,222,0.5);
                    }}
                    button:hover {{
                        box-shadow: 0 0 20px rgba(242,231,222,0.8);
                        background-color: #5C7999; /* Waikawa Gray hover */
                        color: #F2E7DE;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 15px;
                        font-size: 1.3em;
                    }}
                    th, td {{
                        border: 1px solid #A7C2D3; /* Casper border */
                        padding: 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #5C7999; /* Waikawa Gray */
                        color: #F2E7DE;
                    }}
                    td {{
                        background-color: #3B5B8C; /* Chambray */
                        color: #F2E7DE;
                    }}

                    /* Pulsing prediction animation */
                    @keyframes pulse-prediction {{
                        0% {{ background: linear-gradient(90deg, #A7C2D3, #F2E7DE); box-shadow: 0 0 5px #F2E7DE; }}
                        50% {{ background: linear-gradient(90deg, #F2E7DE, #A7C2D3); box-shadow: 0 0 20px #F2E7DE; }}
                        100% {{ background: linear-gradient(90deg, #A7C2D3, #F2E7DE); box-shadow: 0 0 5px #F2E7DE; }}
                    }}
                    .prediction-row td {{
                        font-weight: bold;
                        font-size: 2em;
                        color: #1F4060; /* text Cello */
                        text-align: center;
                        animation: pulse-prediction 2s infinite;
                        border-radius: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <h1>California House Price Prediction</h1>
                        <form method="post">
                            {"".join([f"<label for='{key}'>{label}</label><input type='number' step='0.01' name='{key}' required>" for key, label in feature_names])}
                            <button type="submit">Predict</button>
                        </form>
                    </div>
                    <div class="card">
                        <h2>Prediction</h2>
                        {"<table><tr><th>Feature</th><th>Value</th></tr>" + table_rows + "</table>" if features else "<p style='font-size:1.2em;'>Enter values to see prediction.</p>"}
                    </div>
                </div>
            </body>
        </html>
        """
        return func.HttpResponse(html_content, mimetype="text/html", status_code=200)

    except Exception as e:
        logging.error(f"Result page error: {e}")
        return func.HttpResponse(f"<h3>Error: {str(e)}</h3>", mimetype="text/html", status_code=500)
