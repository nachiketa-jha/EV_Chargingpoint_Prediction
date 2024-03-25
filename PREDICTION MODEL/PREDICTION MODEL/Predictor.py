import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Load preprocessed data from CSV file
delhi_data = pd.read_csv('6th SEM/minor project /PREDICTION MODEL/delhi_data.csv')


# Convert 'Area Type' column to categorical datatype
delhi_data['Area Type'] = pd.Categorical(delhi_data['Area Type'])

# One-hot encode 'Area Type' column
delhi_data = pd.get_dummies(delhi_data, columns=['Area Type'])
# Assuming 'Latitude' and 'Longitude' are the target variables, and all other columns are features
X = delhi_data.drop(columns=['Area Name', 'Latitude', 'Longitude'])
y_lat = delhi_data['Latitude']
y_long = delhi_data['Longitude']

# Split the data into training and testing sets for latitude prediction
X_train_lat, X_test_lat, y_train_lat, y_test_lat = train_test_split(X, y_lat, test_size=0.2, random_state=42)

# Split the data into training and testing sets for longitude prediction
X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(X, y_long, test_size=0.2, random_state=42)


# Initialize Random Forest regressors for latitude and longitude prediction
rf_model_lat = RandomForestRegressor(random_state=42)
rf_model_long = RandomForestRegressor(random_state=42)

# Training for latitude prediction
print("Training for latitude prediction started...")
start_time_lat = time.time()
rf_model_lat.fit(X_train_lat, y_train_lat)
end_time_lat = time.time()
print("Training for latitude prediction completed.")
print("Time taken for training: {:.2f} seconds".format(end_time_lat - start_time_lat))

# Training for longitude prediction
print("\nTraining for longitude prediction started...")
start_time_long = time.time()
rf_model_long.fit(X_train_long, y_train_long)
end_time_long = time.time()
print("Training for longitude prediction completed.")
print("Time taken for training: {:.2f} seconds".format(end_time_long - start_time_long))

# Predictions...
predicted_coordinates = []
for area in delhi_data.itertuples():
    # Extract features for the area
    area_features = np.array(area[4:])  # Exclude 'Area Name', 'Latitude', and 'Longitude'
    area_features = area_features.reshape(1, -1)
    
    # Predict latitude for the area
    area_predicted_latitude = rf_model_lat.predict(area_features)
    
    # Predict longitude for the area
    area_predicted_longitude = rf_model_long.predict(area_features)
    
    # Add predicted coordinates for the area to the list
    predicted_coordinates.append((area[1], area_predicted_latitude[0], area_predicted_longitude[0]))

# Print predicted coordinates for each area
# for coord in predicted_coordinates:
#     print("Area: {}, Latitude: {}, Longitude: {}".format(coord[0], coord[1], coord[2]))

while True:
    # Prompt the user to input the name of the area for which they want to predict the EV charging station location
    area_name_input = input("Enter the name of the area for EV charging station prediction or 'quit' to exit: ")

    if area_name_input.lower() == 'quit':
        print("Exiting program...")
        break

    # Find the row corresponding to the input area name
    area_row = delhi_data[delhi_data['Area Name'] == area_name_input]

    if not area_row.empty:
        # Extract features for the area
        area_features = area_row.drop(columns=['Area Name', 'Latitude', 'Longitude'])

        # Set to store unique predicted coordinates
        unique_coordinates = set()

        # Predict coordinates multiple times for the input area
        for _ in range(5):  # Predict 5 coordinates
            # Predict latitude for the area
            area_predicted_latitude = rf_model_lat.predict(area_features)

            # Predict longitude for the area
            area_predicted_longitude = rf_model_long.predict(area_features)

            # Add predicted coordinates to the set
            unique_coordinates.add((area_predicted_latitude[0], area_predicted_longitude[0]))

        # Print the number of possible unique coordinates
        print("\nNumber of possible EV charging station in {}: {}".format(area_name_input, len(unique_coordinates)))

        # Print predicted coordinates for the input area
        print("Predicted coordinates:")
        for i, (latitude, longitude) in enumerate(unique_coordinates, 1):
            print("Coordinate {}: Latitude: {}, Longitude: {}".format(i, latitude, longitude))
    else:
        print("Area not found in the dataset. Please enter a valid area name or type 'quit' to exit.")

