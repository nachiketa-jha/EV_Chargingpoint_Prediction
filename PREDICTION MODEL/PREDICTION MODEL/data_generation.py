import pandas as pd
import numpy as np

# List of 100 real area names in Delhi
area_names = [
"Connaught Place",
    "Chanakyapuri",
    "Karol Bagh",
    "Dwarka",
    "Preet Vihar",
    "Rohini",
    "Vasant Vihar",
    "Nehru Place",
    "Lajpat Nagar",
    "Mayur Vihar",'Asaf Ali Road', 'Bhikaji Cama Palace', 'Chandni Chowk Old Delhi', 'Chhattarpur', 'Chirag Enclave', 
    'Civil Lines', 'Connaught Place', 'Diplomatic Enclave', 'East of Kailash', 'Friends Colony', 
    'Golf Links', 'Greater Kailash - I', 'Green Park', 'IIT', 'International Airport', 'Karol Bagh', 
    'Lajpat Nagar', 'Mehrauli Gurgaon Road', 'Naraina', 'Nehru Place', 'New Friends Colony', 'Nizammuddin', 
    'Paharganj', 'Panchshil Enclave', 'Patel Nagar', 'Pushpanjali Farms', 'Qutab', 'Rajendra Place', 
    'Rajokri', 'Saket', 'Samalka', 'Shalimar Bagh', 'Shiv Murti', 'Sukhdev Vihar', 'Sundar Nagar', 
    'Vasant Kunj', 'Vasant Vihar', 'Dwarka', 'Defence Colony', 'Shahdara', 'Mayur Vihar', 'Narela', 
    'Paschim Vihar', 'Adarsh Nagar Metro Station', 'AIIMS Metro Station', 'Akshardham Metro Station', 
    'Akshardham Temple Delhi', 'Anand Vihar ISBT Metro Station', 'Anand Vihar Railway Station', 
    'Arjangarh Metro Station', 'Azadpur Metro Station', 'Badarpur Metro Station', 'Barakhamba Road Metro Station', 
    'Birla Temple Karol Bagh', 'Catherdral Church of Redemption', 'Central Secrateriat Metro Station', 
    'Chandani Chowk', 'Chandani Chowk Metro Station', 'Chawri Bazaar Metro Station', 'Chhattarpur Metro Station', 
    'Civil Lines Metro Station', 'Connaught Place', 'Dilshad Garden Metro Station', 'Dwarka Metro Station', 
    'Dwarka Mor Metro Station', 'Dwarka Sector 08 Metro Station', 'Dwarka Sector 09 Metro Station', 
    'Dwarka Sector 10 Metro Station', 'Dwarka Sector 11 Metro Station', 'Dwarka Sector 12 Metro Station', 
    'Dwarka Sector 13 Metro Station', 'Dwarka Sector 14 Metro Station', 'Dwarka Sector 21 Metro Station', 
    'Fortis Escorts Heart Institute', 'Ghitorni Metro Station', 'Govindpuri Metro Station', 'Green Park Metro Station', 
    'GTB Nagar Metro Station', 'Guru Dronacharya Metro Station', 'Gurudwara Bangla Sahib', 'Hauz Khas Metro Station', 
    'Huda City Center Metro Station', 'Humayun Tomb', 'IFFCO Chowk Metro Station', 'INA Metro Station', 
    'Inder Lok Metro Station', 'India Gate', 'Indian Spinal Injuries Centre', 'Indira Gandhi International Airport', 
    'Indraprastha Apollo Hospitals', 'Indrapratha Metro Station', 'ISKCON Temple Delhi', 'Jahangir Puri Metro Station', 
    'Jaipur Golden Hospital', 'Jama Masjid Delhi', 'Janakpuri East Metro Station', 'Janakpuri West Metro Station', 
    'Jangpura Metro Station', 'Jantar Mantar Delhi', 'Jasola Metro Station', 'Jhandewalaan Metro Station', 
    'Jhilmil Metro Station', 'JLN Stadium Metro Station', 'Jor Bagh Metro Station', 'Kailash Colony Metro Station', 
    'Kalka Ji Mandir', 'Kalkaji Mandir Metro Station', 'Kanhiya Nagar Metro Station', 'Karkarduma Metro Station', 
    'Karol Bagh Metro Station', 'Kashmere Gate Metro Station', 'Keshav Puram Metro Station', 'Khan Market Metro Station', 
    'Kohat Enclave Metro Station', 'Lajpat Nagar Metro Station', 'Laxmi Nagar Metro Station', 'Lodhi Gardens', 
    'Lotus Temple', 'Madipur Metro Station', 'Maharaja Agrasen Hospital', 'Malviya Nagar Metro Station', 
    'Mandi House Metro Station', 'Mansarovar Park Metro Station', 'Mayur Vihar Extension Metro Station', 
    'Mayur Vihar Phase 1 Metro Station', 'MG Road Metro Station', 'Modeltown Metro Station', 'Mohan Estate Metro Station', 
    'Moolchand Metro Station', 'Moti Nagar Metro Station', 'Mundka Metro Station', 'Nangloi Metro Station', 
    'Nangloi Railway Metro Station', 'National Heart Institute', 'Nawada Metro Station', 'Nehru Palace Metro Station', 
    'Netaji Subash Place Metro Station', 'New Ashok Nagar Metro Station', 'NEW DELHI (DMRC) Metro Station', 
    'Nirman Vihar Metro Station', 'Okhla Metro Station', 'Paranthewali Galli', 'Paschim Vihar East Metro Station', 
    'Paschim Vihar West Metro Station', 'Patel Chowk Metro Station', 'Patel Nagar Metro Station', 'Peeragarhi Metro Station', 
    'Pitam Pura Metro Station', 'Pragati Maidan Metro Station', 'Pratap Nagar Metro Station', 'Preet Vihar Metro Station', 
    'Pul Bangash Metro Station', 'Punjabi Bagh Metro Station', 'Qutab Minar', 'Qutab Minar Metro Station', 
    'R K Ashram Marg Metro Station', 'Race Course Metro Station', 'Rajdhani Park Metro Station', 
    'Rajendra Place Metro Station', 'Rajouri Garden Metro Station', 'Ramesh Nagar Metro Station', 'Rashtrapati Bhavan', 
    'Red Fort', 'Rithala Metro Station', 'Rohini East Metro Station', 'Rohini West Metro Station', 'Safdarjung Hospital', 
    'Safdarjung Tomb', 'Saket Metro Station', 'Sansad Bhavan', 'Sarita Vihar Metro Station', 'Seelampur Metro Station', 
    'Shadipur Metro Station', 'Shahdara Metro Station', 'Shastri Nagar Metro Station', 'Shastri Park Metro Station', 
    'Shivaji Park Metro Station', 'Sikanderpur Metro Station', 'Sir Ganga Ram Hospital', 'Subash Nagar Metro Station', 
    'Sultanpur Metro Station', 'Supreme Court - New Delhi', 'Surajmal Stadium Metro Station', 'Tagore Garden Metro Station', 
    'Tilak Nagar Metro Station', 'Tis Hazari Metro Station', 'Tughlakabad Metro Station', 'Udhyog Nagar Metro Station'
]
# Generate random latitude and longitude for each area
latitudes = np.random.uniform(28.4, 28.9, len(area_names))
longitudes = np.random.uniform(76.8, 77.3, len(area_names))

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

# Save the data to a CSV file
delhi_data.to_csv('delhi_data.csv', index=False)
print(delhi_data.head())