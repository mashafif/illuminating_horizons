import requests
import math

api_key = 'AIzaSyAMdfrkUpDoIE6GRRUu6HjwW-0bpOc-MCU'

def get_elevation(coords, api_key):
    locations = '|'.join([f"{lat},{lon}" for lat, lon in coords])  # Unpacking only lat and lon
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={api_key}"
    response = requests.get(url)
    result = response.json()
    if result['status'] == 'OK':
        return [res['elevation'] for res in result['results']]
    else:
        raise Exception(f"API Error: {result['status']}")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in meters

def calculate_slope(elev1, elev2, distance):
    return (elev2 - elev1) / distance * 100  # Slope as a percentage

def get_surrounding_points(lat, lon, distance_meters, num_points=8):
    points = []
    angle_increment = 360 / num_points
    earth_radius = 6371e3  # Radius of Earth in meters
    for i in range(num_points):
        angle = math.radians(i * angle_increment)
        delta_lat = distance_meters / earth_radius * (180 / math.pi)
        delta_lon = delta_lat / math.cos(math.radians(lat))
        lat2 = lat + delta_lat * math.cos(angle)
        lon2 = lon + delta_lon * math.sin(angle)
        points.append((lat2, lon2, i * angle_increment))
    return points

def get_slope(lat, lon, api_key, distance_meters=100, num_points=8):
    # Get surrounding points in a circle around the center point
    points = get_surrounding_points(lat, lon, distance_meters, num_points)
    p = [x[0:2] for x in points]
    # Include the center point in the list of points
    all_points = [(lat, lon)] + points
    p = [x[0:2] for x in all_points]
    # Get elevation for all points
    elevations = get_elevation(p, api_key)

    # The first elevation corresponds to the center point
    elev_center = elevations[0]
    steepest_slope = None
    steepest_bearing = None

    for i, (lat2, lon2, bearing) in enumerate(points):
        # Get elevation of the surrounding point
        elev2 = elevations[i + 1]  # +1 because the first elevation is for the center point
        # Calculate horizontal distance between points
        distance = haversine(lat, lon, lat2, lon2)
        # Calculate slope
        slope = calculate_slope(elev_center, elev2, distance)
        if steepest_slope is None or slope < steepest_slope:
            steepest_slope = slope
            steepest_bearing = bearing

    return elev_center, steepest_slope, steepest_bearing

# Example usage
#lat, lon = 34.902871, 139.097548  # Example location
#elevation, steepest_slope, bearing = get_slope(lat, lon, api_key)

#print(f"Elevation: {elevation:.2f} meters")
#print(f"Steepest Slope: {steepest_slope:.2f}%")
#print(f"Bearing (direction): {bearing:.2f}°")
