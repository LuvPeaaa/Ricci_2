import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def load_time_series(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date')
    df = df.set_index('date')
    df = df.select_dtypes(include=np.number)
    return df

# Thêm đoạn debug để xem quá trình tính toán
def build_graph_from_series_optimized(data, k=3, method='cosine'):
    G = nx.Graph()
    dates = data.index.to_list()
    features = data.values
    
    print(f"Building graph with {len(dates)} nodes, method: {method}")
    
    # Add nodes
    for i, date in enumerate(dates):
        G.add_node(i, date=date)
    
    print("Computing distance matrix...")
    # Calculate similarity/distance matrix once
    try:
        if method == 'cosine':
            # Check for zero vectors in advance
            zero_vectors = np.where(np.all(features == 0, axis=1))[0]
            if len(zero_vectors) > 0:
                print(f"Warning: Found {len(zero_vectors)} zero vectors. Setting max distance for them.")
            
            # Compute cosine similarity matrix for all pairs
            similarity_matrix = cosine_similarity(features)
            distance_matrix = 1 - similarity_matrix
        else:
            # Create Euclidean distance matrix
            n = len(features)
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                if i % 100 == 0:  # Progress report for large datasets
                    print(f"Computing Euclidean distances: {i}/{n}")
                for j in range(i+1, n):
                    distance_matrix[i, j] = euclidean(features[i], features[j])
                    distance_matrix[j, i] = distance_matrix[i, j]
    except Exception as e:
        print(f"Error during distance matrix calculation: {e}")
        raise
    
    print("Creating graph edges...")
    # Create graph from distance matrix
    for i in range(len(features)):
        if i % 100 == 0:  # Progress report
            print(f"Processing node {i}/{len(features)}")
        
        try:
            # Get nearest neighbors for node i
            # Exclude distance to itself (=0)
            distances = [(j, distance_matrix[i, j]) for j in range(len(features)) if i != j]
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k]
            for j, d in neighbors:
                G.add_edge(i, j, weight=d)
        except Exception as e:
            print(f"Error processing node {i}: {e}")
    
    print("Graph building complete.")
    return G, dates

def compute_forman_ricci(G):
    fr = FormanRicci(G)
    fr.compute_ricci_curvature()
    return fr.G

def compute_ollivier_ricci(G):
    orc = OllivierRicci(G, alpha=0.5, verbose=False, proc=1)  # Avoid multiprocessing
    orc.compute_ricci_curvature()
    return orc.G

def extract_avg_curvature(G, dates, key='formanCurvature'):
    avg_ricci_by_day = []
    for i in range(len(dates)):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            avg_ricci_by_day.append(0)
            continue
        curvatures = [G[i][j].get(key, 0) for j in neighbors]
        avg = np.mean(curvatures)
        avg_ricci_by_day.append(avg)
    return avg_ricci_by_day

def plot_curvatures(dates, forman, ollivier):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, forman, label='Forman-Ricci', marker='o')
    plt.plot(dates, ollivier, label='Ollivier-Ricci', marker='x')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Ricci Curvature Over Time (Forman vs Ollivier)")
    plt.xlabel("Date")
    plt.ylabel("Average Ricci Curvature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(file_path, k=5, method='cosine'):
    ts = load_time_series(file_path)
    print(f"Loaded time series with shape: {ts.shape}")
    
    # Use the optimized function instead of the original
    G, dates = build_graph_from_series_optimized(ts, k=k, method=method)

    print("[•] Calculating Forman-Ricci...")
    G_forman = compute_forman_ricci(G.copy())
    forman_curve = extract_avg_curvature(G_forman, dates, key='formanCurvature')

    print("[•] Calculating Ollivier-Ricci...")
    G_ollivier = compute_ollivier_ricci(G.copy())
    ollivier_curve = extract_avg_curvature(G_ollivier, dates, key='ricciCurvature')

    plot_curvatures(dates, forman_curve, ollivier_curve)

if __name__ == "__main__":
    # Cập nhật đường dẫn phù hợp
    file_path = "C:\\data\\processed_stock.csv"
    main(file_path)