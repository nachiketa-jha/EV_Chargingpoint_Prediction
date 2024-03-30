import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import time

# Suppress UserWarnings about missing feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# List of 100 real area names in Delhi
# area_names = [
#     "Connaught Place",
#     "Chanakyapuri",
#     "Karol Bagh",
#     "Dwarka",
#     "Preet Vihar",
#     "Rohini",
#     "Vasant Vihar",
#     "Nehru Place",
#     "Lajpat Nagar",
#     "Mayur Vihar",'Asaf Ali Road', 'Bhikaji Cama Palace', 'Chandni Chowk Old Delhi', 'Chhattarpur', 'Chirag Enclave', 
#     'Civil Lines', 'Connaught Place', 'Diplomatic Enclave', 'East of Kailash', 'Friends Colony', 
#     'Golf Links', 'Greater Kailash - I', 'Green Park', 'IIT', 'International Airport', 'Karol Bagh', 
#     'Lajpat Nagar', 'Mehrauli Gurgaon Road', 'Naraina', 'Nehru Place', 'New Friends Colony', 'Nizammuddin', 
#     'Paharganj', 'Panchshil Enclave', 'Patel Nagar', 'Pushpanjali Farms', 'Qutab', 'Rajendra Place', 
#     'Rajokri', 'Saket', 'Samalka', 'Shalimar Bagh', 'Shiv Murti', 'Sukhdev Vihar', 'Sundar Nagar', 
#     'Vasant Kunj', 'Vasant Vihar', 'Dwarka', 'Defence Colony', 'Shahdara', 'Mayur Vihar', 'Narela', 
#     'Paschim Vihar', 'Adarsh Nagar Metro Station', 'AIIMS Metro Station', 'Akshardham Metro Station', 
#     'Akshardham Temple Delhi', 'Anand Vihar ISBT Metro Station', 'Anand Vihar Railway Station', 
#     'Arjangarh Metro Station', 'Azadpur Metro Station', 'Badarpur Metro Station', 'Barakhamba Road Metro Station', 
#     'Birla Temple Karol Bagh', 'Catherdral Church of Redemption', 'Central Secrateriat Metro Station', 
#     'Chandani Chowk', 'Chandani Chowk Metro Station', 'Chawri Bazaar Metro Station', 'Chhattarpur Metro Station', 
#     'Civil Lines Metro Station', 'Connaught Place', 'Dilshad Garden Metro Station', 'Dwarka Metro Station', 
#     'Dwarka Mor Metro Station', 'Dwarka Sector 08 Metro Station', 'Dwarka Sector 09 Metro Station', 
#     'Dwarka Sector 10 Metro Station', 'Dwarka Sector 11 Metro Station', 'Dwarka Sector 12 Metro Station', 
#     'Dwarka Sector 13 Metro Station', 'Dwarka Sector 14 Metro Station', 'Dwarka Sector 21 Metro Station', 
#     'Fortis Escorts Heart Institute', 'Ghitorni Metro Station', 'Govindpuri Metro Station', 'Green Park Metro Station', 
#     'GTB Nagar Metro Station', 'Guru Dronacharya Metro Station', 'Gurudwara Bangla Sahib', 'Hauz Khas Metro Station', 
#     'Huda City Center Metro Station', 'Humayun Tomb', 'IFFCO Chowk Metro Station', 'INA Metro Station', 
#     'Inder Lok Metro Station', 'India Gate', 'Indian Spinal Injuries Centre', 'Indira Gandhi International Airport', 
#     'Indraprastha Apollo Hospitals', 'Indrapratha Metro Station', 'ISKCON Temple Delhi', 'Jahangir Puri Metro Station', 
#     'Jaipur Golden Hospital', 'Jama Masjid Delhi', 'Janakpuri East Metro Station', 'Janakpuri West Metro Station', 
#     'Jangpura Metro Station', 'Jantar Mantar Delhi', 'Jasola Metro Station', 'Jhandewalaan Metro Station', 
#     'Jhilmil Metro Station', 'JLN Stadium Metro Station', 'Jor Bagh Metro Station', 'Kailash Colony Metro Station', 
#     'Kalka Ji Mandir', 'Kalkaji Mandir Metro Station', 'Kanhiya Nagar Metro Station', 'Karkarduma Metro Station', 
#     'Karol Bagh Metro Station', 'Kashmere Gate Metro Station', 'Keshav Puram Metro Station', 'Khan Market Metro Station', 
#     'Kohat Enclave Metro Station', 'Lajpat Nagar Metro Station', 'Laxmi Nagar Metro Station', 'Lodhi Gardens', 
#     'Lotus Temple', 'Madipur Metro Station', 'Maharaja Agrasen Hospital', 'Malviya Nagar Metro Station', 
#     'Mandi House Metro Station', 'Mansarovar Park Metro Station', 'Mayur Vihar Extension Metro Station', 
#     'Mayur Vihar Phase 1 Metro Station', 'MG Road Metro Station', 'Modeltown Metro Station', 'Mohan Estate Metro Station', 
#     'Moolchand Metro Station', 'Moti Nagar Metro Station', 'Mundka Metro Station', 'Nangloi Metro Station', 
#     'Nangloi Railway Metro Station', 'National Heart Institute', 'Nawada Metro Station', 'Nehru Palace Metro Station', 
#     'Netaji Subash Place Metro Station', 'New Ashok Nagar Metro Station', 'NEW DELHI (DMRC) Metro Station', 
#     'Nirman Vihar Metro Station', 'Okhla Metro Station', 'Paranthewali Galli', 'Paschim Vihar East Metro Station', 
#     'Paschim Vihar West Metro Station', 'Patel Chowk Metro Station', 'Patel Nagar Metro Station', 'Peeragarhi Metro Station', 
#     'Pitam Pura Metro Station', 'Pragati Maidan Metro Station', 'Pratap Nagar Metro Station', 'Preet Vihar Metro Station', 
#     'Pul Bangash Metro Station', 'Punjabi Bagh Metro Station', 'Qutab Minar', 'Qutab Minar Metro Station', 
#     'R K Ashram Marg Metro Station', 'Race Course Metro Station', 'Rajdhani Park Metro Station', 
#     'Rajendra Place Metro Station', 'Rajouri Garden Metro Station', 'Ramesh Nagar Metro Station', 'Rashtrapati Bhavan', 
#     'Red Fort', 'Rithala Metro Station', 'Rohini East Metro Station', 'Rohini West Metro Station', 'Safdarjung Hospital', 
#     'Safdarjung Tomb', 'Saket Metro Station', 'Sansad Bhavan', 'Sarita Vihar Metro Station', 'Seelampur Metro Station', 
#     'Shadipur Metro Station', 'Shahdara Metro Station', 'Shastri Nagar Metro Station', 'Shastri Park Metro Station', 
#     'Shivaji Park Metro Station', 'Sikanderpur Metro Station', 'Sir Ganga Ram Hospital', 'Subash Nagar Metro Station', 
#     'Sultanpur Metro Station', 'Supreme Court - New Delhi', 'Surajmal Stadium Metro Station', 'Tagore Garden Metro Station', 
#     'Tilak Nagar Metro Station', 'Tis Hazari Metro Station', 'Tughlakabad Metro Station', 'Udhyog Nagar Metro Station'
#     # Add more area names as needed...
# ]
area_names = pd.read_excel(r'petrol_pump_delhi.xlsx')

# Generate random latitude and longitude for each area
latitudes = area_names.Latitude 
longitudes = area_names.Longitude

# Generate dummy data for the rest of the columns
np.random.seed(0)
population_density = np.random.randint(500, 2000, size=len(area_names))
distance_to_highway = np.random.randint(1, 20, size=len(area_names))
urbanization_level = np.random.randint(1, 5, size=len(area_names))
traffic_level = np.random.randint(1, 6, size=len(area_names))
area_type = np.random.choice(['Residential', 'Commercial', 'Industrial'], size=len(area_names))
ev_usage = np.random.randint(1, 11, size=len(area_names))

# Create DataFrame for the dummy data
dummy_data = pd.DataFrame({
    'Population_Density': population_density,
    'Distance_to_Highway': distance_to_highway,
    'Urbanization_Level': urbanization_level,
    'Traffic_Level': traffic_level,
    'Area Type': area_type,
    'EV_Usage': ev_usage
})

# Combine real area names, latitude, and longitude with dummy data
delhi_areas_data = pd.DataFrame({
    'Area Name': area_names,
    'Latitude': latitudes,
    'Longitude': longitudes
})

# Merge real area data with dummy data
delhi_data = pd.concat([delhi_areas_data, dummy_data], axis=1)

# Convert 'Area Type' column to categorical datatype
delhi_data['Area Type'] = pd.Categorical(delhi_data['Area Type'])

# One-hot encode 'Area Type' column
delhi_data = pd.get_dummies(delhi_data, columns=['Area Type'])

# Split the data into features (X) and target variables (latitude and longitude)
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

# Predict latitude and longitude for the new location along a specified route
# Modify this section to predict coordinates along the route
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
