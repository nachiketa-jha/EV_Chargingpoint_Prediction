import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from CSV file
delhi_data = pd.read_csv('6th SEM/minor project /PREDICTION MODEL/delhi_data.csv')

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
