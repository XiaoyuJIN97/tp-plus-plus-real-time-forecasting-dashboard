"""Original DE solar point-selection algorithm extracted from the internal v0.1 notebook.

Requires the ORIGINAL renewables_location_all.csv, not the newer MaStR inventory.
Outputs the original reference / ERA5 / nominal GFS coordinates. It does not
replace the collector's currently configured request coordinates.
"""
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd

N_POINTS = 4
ERA5_GRID_DEGREES = 0.25
GFS_GRID_DEGREES = 0.1171875

def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def neighboring_grid_values(value, grid_degrees=ERA5_GRID_DEGREES):
    scaled = float(value) / grid_degrees
    low = math.floor(scaled) * grid_degrees
    high = math.ceil(scaled) * grid_degrees
    return sorted({round(low, 6), round(high, 6)})


def nearest_grid_point(lat, lon, grid_degrees):
    candidates = []
    for candidate_lat in neighboring_grid_values(lat, grid_degrees):
        for candidate_lon in neighboring_grid_values(lon, grid_degrees):
            candidates.append((haversine_km(lat, lon, candidate_lat, candidate_lon), candidate_lat, candidate_lon))
    _, best_lat, best_lon = min(candidates, key=lambda item: item[0])
    return best_lat, best_lon


def snap_to_era5_grid(lat, lon):
    return nearest_grid_point(lat, lon, ERA5_GRID_DEGREES)


def nearest_unused_era5_grid_point(lat, lon, used_keys, max_steps=4):
    # First try the nearest cell. If it is already used for this country/type,
    # expand around it until the closest unused ERA5 cell is found.
    base_lat, base_lon = snap_to_era5_grid(lat, lon)
    candidates = []
    for step in range(max_steps + 1):
        offsets = range(-step, step + 1)
        for i in offsets:
            for j in offsets:
                candidate_lat = round(base_lat + i * ERA5_GRID_DEGREES, 6)
                candidate_lon = round(base_lon + j * ERA5_GRID_DEGREES, 6)
                key = (round(candidate_lat, 3), round(candidate_lon, 3))
                if key in used_keys:
                    continue
                candidates.append((haversine_km(lat, lon, candidate_lat, candidate_lon), candidate_lat, candidate_lon))
        if candidates:
            _, best_lat, best_lon = min(candidates, key=lambda item: item[0])
            return best_lat, best_lon
    return base_lat, base_lon


def nearest_gfs_grid_point(lat, lon):
    return nearest_grid_point(lat, lon, GFS_GRID_DEGREES)


def capacity_weighted_centroid(frame, weight_col='capacity'):
    weights = pd.to_numeric(frame[weight_col], errors='coerce').fillna(0.0)
    if weights.sum() <= 0:
        return frame['lat'].mean(), frame['lon'].mean()
    return np.average(frame['lat'], weights=weights), np.average(frame['lon'], weights=weights)


def weighted_kmeans_points(frame, k=N_POINTS, weight_col='capacity', max_iter=100):
    data = frame[['lat', 'lon']].astype(float).to_numpy()
    weights = pd.to_numeric(frame[weight_col], errors='coerce').fillna(0.0).to_numpy()
    if len(frame) == 0:
        raise ValueError('Cannot cluster an empty frame.')
    if weights.sum() <= 0:
        weights = np.ones(len(frame))

    unique = pd.DataFrame(data, columns=['lat', 'lon']).drop_duplicates().to_numpy()
    if len(unique) >= k:
        first_idx = int(np.argmax(weights))
        centers = [data[first_idx]]
        while len(centers) < k:
            distances = np.array([min(np.sum((point - center) ** 2) for center in centers) for point in data])
            next_idx = int(np.argmax(distances * weights))
            centers.append(data[next_idx])
        centers = np.array(centers, dtype=float)
    else:
        lat, lon = capacity_weighted_centroid(frame, weight_col)
        offsets = [(0, 0), (0.15, 0), (-0.15, 0), (0, 0.15), (0, -0.15), (0.15, 0.15)]
        centers = np.array([[lat + offsets[i][0], lon + offsets[i][1]] for i in range(k)], dtype=float)

    labels = np.zeros(len(data), dtype=int)
    for _ in range(max_iter):
        distances = np.stack([np.sum((data - center) ** 2, axis=1) for center in centers], axis=1)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            mask = new_labels == cluster_id
            if mask.any():
                cluster_weights = weights[mask]
                if cluster_weights.sum() <= 0:
                    new_centers[cluster_id] = data[mask].mean(axis=0)
                else:
                    new_centers[cluster_id] = np.average(data[mask], axis=0, weights=cluster_weights)
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels = new_labels
        centers = new_centers

    rows = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster = frame.iloc[np.where(mask)[0]].copy()
        if cluster.empty:
            lat, lon = centers[cluster_id]
            capacity = 0.0
            assets = 0
        else:
            lat, lon = capacity_weighted_centroid(cluster, weight_col)
            capacity = pd.to_numeric(cluster[weight_col], errors='coerce').fillna(0.0).sum()
            assets = len(cluster)
        rows.append({'cluster_id': cluster_id + 1, 'reference_lat': lat, 'reference_lon': lon, 'cluster_assets': assets, 'cluster_capacity_mw': capacity})
    return rows


def select_de_solar(metadata):
    assets = pd.read_csv(metadata).rename(columns={'capacity_mw':'capacity'})
    assets['country'] = assets.country.replace({'DE-LU':'DE'})
    assets['target_type'] = assets['type'].map({'solar':'solar','onshore_wind':'onshore','offshore_wind':'offshore'})
    for name in ('capacity','lat','lon'):
        assets[name] = pd.to_numeric(assets[name], errors='coerce')
    assets['capacity'] = assets.capacity.fillna(0.)
    assets = assets.dropna(subset=['country','target_type','lat','lon'])
    assets = assets[assets.country.eq('DE') & assets.target_type.eq('solar')].copy()
    rows = []; used = set()
    for point, ref in enumerate(weighted_kmeans_points(assets), start=1):
        lat,lon = nearest_unused_era5_grid_point(ref['reference_lat'], ref['reference_lon'], used)
        used.add((round(lat,3),round(lon,3)))
        glat,glon = nearest_gfs_grid_point(lat,lon)
        rows.append(dict(country='DE',type='solar',point=point,**ref,
                         era5_lat=round(lat,3),era5_lon=round(lon,3),gfs_lat=round(glat,6),gfs_lon=round(glon,6),
                         selection_method='capacity_weighted_kmeans_solar',
                         distance_reference_era5_km=haversine_km(ref['reference_lat'],ref['reference_lon'],lat,lon),
                         distance_gfs_era5_km=haversine_km(lat,lon,glat,glon)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    select_de_solar(args.metadata).to_csv(args.output,index=False)
