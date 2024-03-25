import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import osmnx as ox
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\nachi\Desktop\petrol_pump_delhi.xlsx")

df.drop_duplicates(subset=["Address"], inplace=True)
df.replace("Not Available", pd.np.nan, inplace=True)
scaler = StandardScaler()
df[["latitude", "longitude"]] = scaler.fit_transform(df[["Latitude", "Longitude"]])
df = pd.get_dummies(df, columns=["Phone_office", "Zone_name"])

location = 'Delhi, India'
# Get the road network for the specified area
G = ox.graph_from_place(location, network_type='drive')
# Project the graph to UTM (Universal Transverse Mercator) for accurate distance calculations
G = ox.project_graph(G)

nodes, edges = ox.graph_to_gdfs(G)

geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
crs = edges.crs  # Get the CRS (Coordinate Reference System) from the edges GeoDataFrame
charging_stations = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

charging_station_nodes = []
for geometry in charging_stations['geometry']:
    # Get the nearest node to the charging station
    nearest_node = ox.nearest_nodes(G, geometry.x, geometry.y)
    
    # Append the nearest node to the list
    charging_station_nodes.append(nearest_node)

for i, (idx, station) in enumerate(charging_stations.iterrows()):
    node_id = charging_station_nodes[i]
    G.nodes[node_id]['charging_station'] = True
    G.nodes[node_id]['name'] = station['name']
    G.nodes[node_id]['latitude'] = station['latitude']
    G.nodes[node_id]['longitude'] = station['longitude']

ox.save_graphml(G, 'delhi_ev_graph.graphml')


fig, ax = ox.plot_graph(G, node_color='r', node_size=5, bgcolor='w', edge_linewidth=0.5, edge_alpha=0.5, show=False, close=False)

# Add charging station locations to the map
charging_stations.plot(ax=ax, markersize=50, marker='o', color='blue', zorder=3)

# Set title and display the map
ax.set_title("Delhi EV Charging Stations \n possibilities")
print(plt.show())