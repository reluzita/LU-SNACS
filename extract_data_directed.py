import networkx as nx
import time
import pandas as pd
from tqdm import tqdm
import sys
from math import log


def common_neighbors(G, node_1, node_2):
    return set(G[node_1]).intersection(G[node_2])


def adamic_adar(G, common):
    return sum(1 / log(G.degree(w)) for w in common)


def jaccard(G, features):
    union = len(set(G[features.name[0]]).union(G[features.name[1]]))
    return 0 if union == 0 else features["common_neighbors"] / union


def get_features(args):
    print(f"Extracting {args[1]}...")
    value = args[0].index.map(args[2])
    print(f"Finished {args[1]}.")
    return args[1], value


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_data_directed.py <dataset> <n>")
        sys.exit(1)

    dataset = sys.argv[1]
    n_deg = int(sys.argv[2])

    edges = []
    with open(f"data/datasets/{dataset}/out.{dataset}", "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(" ")
            if data[1].find("\t") != -1:
                data = [data[0]] + data[1].split("\t")
            edges.append((int(data[3]), (int(data[0]), int(data[1]), int(data[2]))))

    edges = sorted(edges, key=lambda x: x[0], reverse=False)
    feature_edges = [e[1] for e in edges[:int(len(edges)*0.7)]]
    label_edges = {(e[1][0], e[1][1]): 1 for e in edges[int(len(edges)*0.7):]}

    G = nx.DiGraph()
    G.add_weighted_edges_from(feature_edges)
    G_und = G.to_undirected()
    
    print(f"Extracting the {n_deg}-degree neighbors of each node...")

    nodes = set()
    for edge in label_edges.keys():
        nodes.add(edge[0])
        nodes.add(edge[1])

    pairs = []
    for node in tqdm(nodes):
        if node in G.nodes:
            pairs.extend([(node, k_n) for k_n in nx.descendants_at_distance(G, node, n_deg)])

    print(f"Extracted {len(pairs)} pairs of nodes.")

    start = time.time()
    print("Extracting features...")

    print("Calculating katz...")
    katz_dict = nx.katz_centrality(G)

    features_dict = {
        "indegree_i": lambda p: G.in_degree(p[0]),
        "indegree_j": lambda p: G.in_degree(p[1]),
        "outdegree_i": lambda p: G.out_degree(p[0]),
        "outdegree_j": lambda p: G.out_degree(p[1]),
        "katz_i": lambda p: katz_dict.get(p[0], 0),
        "katz_j": lambda p: katz_dict.get(p[1], 0),
        "common_neighbors_list": lambda p: common_neighbors(G_und, p[0], p[1]),
        "pref_attach": lambda p: list(nx.preferential_attachment(G_und, [(p[0], p[1])]))[0][2],
        "label": lambda p: label_edges.get(p, 0),
    }

    features = pd.DataFrame(index=pairs)

    for feature_name, func in features_dict.items():
        print(f"Extracting {feature_name}...")
        features[feature_name] = features.index.map(func)

    features["common_neighbors"] = features["common_neighbors_list"].map(len)
    print("Extracting adamic_adar...")
    features["adamic_adar"] = features["common_neighbors_list"].map(lambda c: adamic_adar(G_und, c))
    print("Extracting jaccard...")
    features["jaccard"] = features.apply(lambda row: jaccard(G_und, row), axis=1)

    features.drop("common_neighbors_list")

    print(f"Took {time.time() - start} seconds to get features")

    print("Writing to file...")

    features.to_csv(f"data/clean_datasets/{dataset}_{n_deg}.csv")

    print("Done!")