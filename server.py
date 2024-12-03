from flask import Flask, request, jsonify,render_template
import pandas as pd
import joblib
import pickle
import base64
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)

# Load the model and grid data
model = joblib.load("model.pkl")
grid_data = pd.read_csv("datagrid.csv")
# If your model requires specific features, ensure they're computed

@app.route('/')
def hello_world():
    return render_template("index.html")


# Function to filter data by location
def filter_by_location(grid_data, location_name):
    geolocator = Nominatim(user_agent="location_filter")
    location = geolocator.geocode(location_name, exactly_one=True, country_codes="DK")
    if not location:
        raise ValueError(f"Location '{location_name}' not found in Denmark.")

    min_lat, max_lat = float(location.raw["boundingbox"][0]), float(location.raw["boundingbox"][1])
    min_lon, max_lon = float(location.raw["boundingbox"][2]), float(location.raw["boundingbox"][3])

    return grid_data[
        (grid_data["latitude"] >= min_lat) & 
        (grid_data["latitude"] <= max_lat) & 
        (grid_data["longitude"] >= min_lon) & 
        (grid_data["longitude"] <= max_lon)
    ]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() # This gets the raw JSON data
    print("Received data:", data)
    plant = data["plant"]
    location = data["location"]
    int_features=[int(x)for x in request. form.values()]
    final=[np.array(int_features)]
    try:
        # Filter data by location
        filtered_data = filter_by_location(grid_data, location)
        
        if filtered_data.empty:
            return jsonify({"error": "No data for this location"}), 400
        
        # Inspect the data shape for debugging purposes
        print(f"Filtered data shape: {filtered_data.shape}")
        
        # Convert categorical columns to numeric (if needed)
        categorical_columns = ['soilType']  # Add other categorical columns if needed
        for col in categorical_columns:
            if filtered_data[col].dtype == 'object':
                label_encoder = LabelEncoder()  # Create a new encoder if not already created
                filtered_data[col] = label_encoder.fit_transform(filtered_data[col])

        # Prepare features for prediction
        # Make sure only the relevant columns are used (e.g., excluding latitude, longitude, and label)
        #features = filtered_data[["feature1", "feature2"]].values  # Replace with actual feature names used in training
       # Replace 'feature1' and 'feature2' with actual column names
        filtered_data['feature1'] = filtered_data['latitude']
        filtered_data['feature2'] = filtered_data['longitude']

        features = filtered_data[["feature1", "feature2"]].values  # Replace with actual feature names used in training
       
        # Inspect the features shape before passing to the model
        print(f"Features shape: {features.shape}")

        # Ensure the number of features matches what the model expects
        if features.shape[1] != 2:  # 2 features expected by LabelPropagation model
            return jsonify({"error": f"Expected 2 features, but got {features.shape[1]}"}), 400
        
        # Make prediction
       
        prediction=model.predict_proba(final)
        output='{0:.{1}f}'.format(prediction[0][1],2)
        probabilities = model.predict_proba(features)[:, 1]
        #prediction_percentage = round((probabilities > 0.5).mean() * 100, 2)

        # Prepare heatmap data
        heatmap_data = [
            [lat, lon, prob]
            for (lat, lon), prob in zip(
                filtered_data[["latitude", "longitude"]].values, probabilities
            )
        ]

        return jsonify(
            {
                "heatmapData": heatmap_data,
                "predictionPercentage": output,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)