import argparse
import networkx as nx
import time
from network_features import common_neighbors, adamic_adar, jaccard
import pandas as pd
from tqdm import tqdm 
from networkx.algorithms import bipartite

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Extract data',
        description = 'Construct dataset for supervised link prediction from a bipartite network dataset')

    parser.add_argument('n', type=int, help='Degree of the neighborhood to extract', choices=[3, 5])

    args = parser.parse_args()
    n_deg = args.n

    edges_dict = {}
    user_nodes = set()
    tag_nodes = set()
    with open('data/datasets/munmun_twitterex_ut/out.munmun_twitterex_ut', "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(" ")
            if int(data[0]) <= 1000 and int(data[1]) <= 10000:
                user = 'user' + data[0]
                tag = 'tag' + data[1]
                user_nodes.add(user)
                tag_nodes.add(tag)
                
                if (user, tag) not in edges_dict:
                    edges_dict[(user, tag)] = {
                        'weight': 1,
                        'timestamp': float(data[3])}
                else:
                    edges_dict[(user, tag)]['weight'] += 1
            
    edges = [(v['timestamp'], (k[0], k[1], v['weight'])) for k, v in edges_dict.items()]
    edges = sorted(edges, key=lambda x: x[0], reverse=False)

    feature_edges = [e[1] for e in edges[:int(len(edges)*0.7)]]
    label_edges = {(e[1][0], e[1][1]): 1 for e in edges[int(len(edges)*0.7):]}

    B = nx.Graph()
    # B.add_nodes_from(user_nodes, bipartite=0)
    # B.add_nodes_from(tag_nodes, bipartite=1)
    B.add_weighted_edges_from(feature_edges)

    nodes = set(edge[0] for edge in label_edges.keys())

    print(f"Extracting the {n_deg}-degree neighbors of each node...")

    pairs = []
    for node in tqdm(nodes):
        if node in B.nodes:
            pairs.extend([(node, k_n) for k_n in nx.descendants_at_distance(B, node, n_deg)])

    start = time.time()
    print("Extracting features...")

    features_dict = {
        "degree_i": lambda p: B.degree(p[0]),
        "degree_j": lambda p: B.degree(p[1]),
        "volume_i": lambda p: nx.volume(B, [p[0]]),
        "volume_j": lambda p: nx.volume(B, [p[1]])
    }
    
    print("Calculating katz...")
    katz_dict = nx.katz_centrality_numpy(B)

    features_dict["katz_i"] = lambda p: katz_dict.get(p[0], 0)
    features_dict["katz_j"] = lambda p: katz_dict.get(p[1], 0)

    features_dict["pref_attach"] = lambda p: list(nx.preferential_attachment(B, [(p[0], p[1])]))[0][2]

    print("Calculating pagerank...")
    pagerank_dict = nx.pagerank(B)

    features_dict["pagerank_i"] = lambda p: pagerank_dict[p[0]]
    features_dict["pagerank_j"] = lambda p: pagerank_dict[p[1]]

    features_dict["label"] = lambda p: label_edges.get(p, 0)
    features = pd.DataFrame(index=pairs)

    for feature_name, func in features_dict.items():
        print(f"Extracting {feature_name}...")
        features[feature_name] = features.index.map(func)


    print(f"Took {time.time() - start} seconds to get features")

    print("Writing to file...")
    features.to_csv(f"data/clean_datasets/munmun_twitterex_ut_{n_deg}.csv")

    print("Done!")