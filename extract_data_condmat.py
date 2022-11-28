import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_data_condmat.py <data_file> <n>")
        sys.exit(1)
        
    data_file = sys.argv[1]
    G = nx.read_adjlist('data/' + data_file, create_using=nx.DiGraph())

    node = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
    nodes = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())
    G0 = G.subgraph(nodes)

    data = random.sample(G0.edges, round(G0.number_of_edges()*0.3))
    G0 = nx.DiGraph(G0)
    G0.remove_edges_from(data)

    n_degree = int(sys.argv[2])
    print(f"Extracting the {n_degree}-degree neighbors of each node...")
    pairs = []
    for node in tqdm(G0.nodes):
        kneighbors = nx.single_source_shortest_path_length(G0, node, cutoff=n_degree)
        for n in kneighbors:
            if kneighbors[n] == n_degree and (n, node) not in pairs:
                pairs.append((node, n))
    
    G0_und = nx.Graph(G0)

    indegree_i = []
    outdegree_i = []
    indegree_j = []
    outdegree_j = []
    common_neighbors = []
    #maxflow = []
    shortest_path = []
    #katz = []
    adamic_adar = []
    pref_attach = []
    jaccard = []

    print("Extracting features...")
    for pair in tqdm(pairs):
        indegree_i.append(G0.in_degree(pair[0]))
        outdegree_i.append(G0.out_degree(pair[0]))
        indegree_j.append(G0.in_degree(pair[1]))
        outdegree_j.append(G0.out_degree(pair[1]))
        common_neighbors.append(len(list(nx.common_neighbors(G0_und, pair[0],pair[1]))))
        shortest_path.append(nx.shortest_path_length(G0, pair[0], pair[1]))
        adamic_adar.append(list(nx.adamic_adar_index(G0_und, [(pair[0], pair[1])]))[0][2])
        pref_attach.append(list(nx.preferential_attachment(G0_und, [(pair[0], pair[1])]))[0][2])
        jaccard.append(list(nx.jaccard_coefficient(G0_und, [(pair[0], pair[1])]))[0][2])

    features = pd.DataFrame({
        'indegree_i': indegree_i,
        'outdegree_i': outdegree_i,
        'indegree_j': indegree_j,
        'outdegree_j': outdegree_j,
        'common_neighbors': common_neighbors,
        'shortest_path': shortest_path,
        'adamic_adar': adamic_adar,
        'pref_attach': pref_attach,
        'jaccard': jaccard
    },columns=['indegree_i', 'outdegree_i', 'indegree_j', 'outdegree_j', 'common_neighbors', 'shortest_path',
    'adamic_adar', 'pref_attach', 'jaccard'], index=pairs)

    print("Determining labels...")
    label = []
    for i, row in features.iterrows():
        if i in data or (i[1], i[0]) in data:
            label.append(1)
        else:
            label.append(0)
    features['label'] = label

    filename = data_file.split('.')[0]
    features.to_csv(f"data/{filename}_n{sys.argv[2]}.csv")