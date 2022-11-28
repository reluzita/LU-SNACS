import networkx as nx
import pandas as pd
from tqdm import tqdm
import sys

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
    label_edges = [(e[1][0], e[1][1]) for e in edges[int(len(edges)*0.7):]]

    G = nx.DiGraph()
    G.add_weighted_edges_from(feature_edges)
    G_und = G.to_undirected()
    
    print(f"Extracting the {n_deg}-degree neighbors of each node...")

    nodes = set()
    for edge in label_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])

    pairs = []
    if n_deg == 2:
        for node in tqdm(nodes):
            if node in G.nodes:
                neighbors = G.neighbors(node)
                for n in neighbors:
                    d2_neighbors = G.neighbors(n)
                    for d2 in [i for i in d2_neighbors if i not in neighbors]:
                        if d2 != node:
                            pairs.append((node, d2))
    else:
        for node in tqdm(nodes):
            kneighbors = nx.single_source_shortest_path_length(G, node, cutoff=n_deg)
            for n in kneighbors:
                if kneighbors[n] == n_deg and (n, node) not in pairs:
                    pairs.append((node, n))

    
    print("Extracting features...")

    indegree_i = []
    indegree_j = []
    outdegree_i = []
    outdegree_j = []
    common_neighbors = []
    adamic_adar = []
    pref_attach = []
    jaccard = []
    label = []

    for pair in tqdm(pairs):
        indegree_i.append(G.in_degree(pair[0]))
        indegree_j.append(G.in_degree(pair[1]))
        outdegree_i.append(G.out_degree(pair[0]))
        outdegree_j.append(G.out_degree(pair[1]))
        common_neighbors.append(len(list(nx.common_neighbors(G_und, pair[0],pair[1]))))
        adamic_adar.append(list(nx.adamic_adar_index(G_und, [(pair[0], pair[1])]))[0][2])
        pref_attach.append(list(nx.preferential_attachment(G_und, [(pair[0], pair[1])]))[0][2])
        jaccard.append(list(nx.jaccard_coefficient(G_und, [(pair[0], pair[1])]))[0][2])
        if pair in label_edges:
            label.append(1)
        else:
            label.append(0)

    features = pd.DataFrame({
        'indegree_i': indegree_i,
        'outdegree_i': outdegree_i,
        'indegree_j': indegree_j,
        'outdegree_j': outdegree_j,
        'common_neighbors': common_neighbors,
        'adamic_adar': adamic_adar,
        'pref_attach': pref_attach,
        'jaccard': jaccard,
        'label': label
    },columns=['indegree_i', 'outdegree_i', 'indegree_j', 'outdegree_j','common_neighbors',
    'adamic_adar', 'pref_attach', 'jaccard', 'label'], index=pairs)

    features.to_csv(f"data/clean_datasets/{dataset}_{n_deg}.csv")

    print("Done!")