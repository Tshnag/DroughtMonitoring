import folium
import pandas as pd
import joblib

# Load your trained machine learning model
model = joblib.load('trained_mlr_model.pkl')

# Load your dataset with the proper encoding (if needed)
data = pd.read_csv('rainfall_area-wt_sd_1901-20151.csv', delimiter=',', encoding='latin-1')
data.dropna(inplace=True)
# Create a map centered on India
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Centered on India

# Loop through the dataset and add markers with drought information and predictions
for index, row in data.iterrows():
    if pd.notna(row['LATITUDE']) and pd.notna(row['LONGITUDE']):
        subdivision = row['SUBDIVISION'].strip()
        latitude = float(row['LATITUDE'].rstrip('°N'))
        longitude = float(row['LONGITUDE'].rstrip('°E'))

        # Make a prediction using your machine learning model
        coordinates=[[latitude,longitude]]
        prediction = model.predict(coordinates)[0]

        # Add a marker with the subdivision name as a tooltip and prediction as a popup
        folium.Marker([latitude, longitude], tooltip=subdivision, popup=f'Drought Prediction: {prediction}').add_to(m)

# Save the map to an HTML file
m.save('drought_prediction_map1.html')
