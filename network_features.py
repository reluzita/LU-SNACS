import time
import networkx as nx
import pandas as pd
from tqdm import tqdm
from math import log

def common_neighbors(G, node_1, node_2):
    return set(G[node_1]).intersection(G[node_2])


def adamic_adar(G, common):
    return sum(1 / log(G.degree(w)) for w in common)


def jaccard(G, features):
    union = len(set(G[features.name[0]]).union(G[features.name[1]]))
    return 0 if union == 0 else features["common_neighbors"] / union

def get_node_pairs(G, label_edges, n_deg):
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

    return pairs

def get_features_dict(G, is_weighted, label_edges, G_und=None):
    features_dict = {}
    if G_und is None:
        features_dict["degree_i"] = lambda p: G.degree(p[0])
        features_dict["degree_j"] = lambda p: G.degree(p[1])
        if is_weighted:
            features_dict["volume_i"] = lambda p: nx.volume(G, [p[0]])
            features_dict["volume_j"] = lambda p: nx.volume(G, [p[1]])

    else:
        features_dict["indegree_i"] = lambda p: G.in_degree(p[0])
        features_dict["indegree_j"] = lambda p: G.in_degree(p[1])
        features_dict["outdegree_i"] = lambda p: G.out_degree(p[0])
        features_dict["outdegree_j"] = lambda p: G.out_degree(p[1])
        if is_weighted:
            features_dict["involume_i"] = lambda p: sum(e[2]['weight'] for e in G.in_edges(p[0], data=True))
            features_dict["outvolume_i"] = lambda p: nx.volume(G, [p[0]])
            features_dict["involume_j"] = lambda p: sum(e[2]['weight'] for e in G.in_edges(p[1], data=True))
            features_dict["outvolume_j"] = lambda p: nx.volume(G, [p[1]])

    # if is_weighted:
    #     features_dict["shortest_path"] = lambda p: nx.shortest_path_length(G, p[0], p[1], weight="weight")
    
    print("Calculating katz...")
    katz_dict = nx.katz_centrality(G)

    features_dict["katz_i"] = lambda p: katz_dict.get(p[0], 0)
    features_dict["katz_j"] = lambda p: katz_dict.get(p[1], 0)

    if G_und:
        features_dict["common_neighbors_list"] = lambda p: common_neighbors(G_und, p[0], p[1])
        features_dict["pref_attach"] = lambda p: list(nx.preferential_attachment(G_und, [(p[0], p[1])]))[0][2]
    else:
        features_dict["common_neighbors_list"] = lambda p: common_neighbors(G, p[0], p[1])
        features_dict["pref_attach"] = lambda p: list(nx.preferential_attachment(G, [(p[0], p[1])]))[0][2]

    print("Calculating pagerank...")
    pagerank_dict = nx.pagerank(G)

    features_dict["pagerank_i"] = lambda p: pagerank_dict[p[0]]
    features_dict["pagerank_j"] = lambda p: pagerank_dict[p[1]]

    features_dict["label"] = lambda p: label_edges.get(p, 0)

    return features_dict


def get_directed_features(feature_edges, label_edges, n_deg, is_weighted):
    G = nx.DiGraph()
    if is_weighted:
        G.add_weighted_edges_from(feature_edges)
    else:
        G.add_edges_from(feature_edges)
    G_und = G.to_undirected()

    pairs = get_node_pairs(G, label_edges, n_deg)

    start = time.time()
    print("Extracting features...")

    features_dict = get_features_dict(G, is_weighted, label_edges, G_und)
    features = pd.DataFrame(index=pairs)

    for feature_name, func in features_dict.items():
        print(f"Extracting {feature_name}...")
        features[feature_name] = features.index.map(func)

    features["common_neighbors"] = features["common_neighbors_list"].map(len)
    print("Extracting adamic_adar...")
    features["adamic_adar"] = features["common_neighbors_list"].map(lambda c: adamic_adar(G_und, c))
    print("Extracting jaccard...")
    features["jaccard"] = features.apply(lambda row: jaccard(G_und, row), axis=1)
    features.drop("common_neighbors_list", axis=1, inplace=True)

    print(f"Took {time.time() - start} seconds to get features")

    return features

def get_undirected_features(feature_edges, label_edges, n_deg, is_weighted):
    G = nx.Graph()
    if is_weighted:
        G.add_weighted_edges_from(feature_edges)
    else:
        G.add_edges_from(feature_edges)

    pairs = get_node_pairs(G, label_edges, n_deg)

    start = time.time()
    print("Extracting features...")

    features_dict = get_features_dict(G, is_weighted, label_edges)
    features = pd.DataFrame(index=pairs)

    for feature_name, func in features_dict.items():
        print(f"Extracting {feature_name}...")
        features[feature_name] = features.index.map(func)

    features["common_neighbors"] = features["common_neighbors_list"].map(len)
    print("Extracting adamic_adar...")
    features["adamic_adar"] = features["common_neighbors_list"].map(lambda c: adamic_adar(G, c))
    print("Extracting jaccard...")
    features["jaccard"] = features.apply(lambda row: jaccard(G, row), axis=1)
    features.drop("common_neighbors_list", axis=1, inplace=True)

    print(f"Took {time.time() - start} seconds to get features")

    return features