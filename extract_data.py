import argparse
from network_features import get_directed_features, get_undirected_features

def read_unweighted_edges(filename):
    edges = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(" ")
            if data[1].find("\t") != -1:
                data = [data[0]] + data[1].split("\t")

            edges.append((int(data[3]), (int(data[0]), int(data[1]))))

    return sorted(edges, key=lambda x: x[0], reverse=False)

def read_weighted_edges(filename):
    edges_dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(" ")
            if data[1].find("\t") != -1:
                data = [data[0]] + data[1].split("\t")
            
            if (int(data[0]), int(data[1])) not in edges_dict:
                edges_dict[(int(data[0]), int(data[1]))] = {
                    'weight': 1,
                    'timestamp': float(data[3])}
            else:
                edges_dict[(int(data[0]), int(data[1]))]['weight'] += 1
            
    edges = [(v['timestamp'], (k[0], k[1], v['weight'])) for k, v in edges_dict.items()]
    return sorted(edges, key=lambda x: x[0], reverse=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'extract_data.py',
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

    if is_weighted:
        edges = read_weighted_edges(f"data/datasets/{dataset}/out.{dataset}")
    else:
        edges = read_unweighted_edges(f"data/datasets/{dataset}/out.{dataset}")
    feature_edges = [e[1] for e in edges[:int(len(edges)*0.7)]]
    label_edges = {(e[1][0], e[1][1]): 1 for e in edges[int(len(edges)*0.7):]}

    if is_directed:
        features = get_directed_features(feature_edges, label_edges, n_deg, is_weighted)
    else:
        features = get_undirected_features(feature_edges, label_edges, n_deg, is_weighted)

    print("Writing to file...")
    features.to_csv(f"data/clean_datasets/{dataset}_{n_deg}.csv")

    print("Done!")