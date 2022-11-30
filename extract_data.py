import networkx as nx
import pandas as pd
from tqdm import tqdm
import argparse
from network_features import get_directed_features, get_undirected_features

def read_edges(filename, is_weighted):
    edges = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(" ")
            if data[1].find("\t") != -1:
                data = [data[0]] + data[1].split("\t")
            
            if is_weighted:
                edges.append((int(data[3]), (int(data[0]), int(data[1]), int(data[2]))))
            else:
                edges.append((int(data[2]), (int(data[0]), int(data[1]))))

    return sorted(edges, key=lambda x: x[0], reverse=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Extract data',
                    description = 'Construct dataset for supervised link prediction from a network dataset')

    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('networktype', type=str, help='Type of dataset (directed or undirected)', choices=['directed', 'undirected'])
    parser.add_argument('edgetype', type=str, help='Type of edges (weighted or unweighted)', choices=['weighted', 'unweighted'])
    parser.add_argument('n', type=int, help='Degree of the neighborhood to extract', choices=[2, 3, 4])

    args = parser.parse_args()

    dataset = args.dataset
    is_directed = (args.networktype == 'directed')
    is_weighted = (args.edgetype == 'weighted')
    n_deg = args.n

    edges = read_edges(f"data/datasets/{dataset}/out.{dataset}", is_weighted)
    feature_edges = [e[1] for e in edges[:int(len(edges)*0.7)]]
    label_edges = [(e[1][0], e[1][1]) for e in edges[int(len(edges)*0.7):]]

    if is_directed:
        features = get_directed_features(feature_edges, label_edges, n_deg)
    else:
        features = get_undirected_features(feature_edges, label_edges, n_deg)

    print("Writing to file...")
    features.to_csv(f"data/clean_datasets/{dataset}_{n_deg}.csv")

    print("Done!")