import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
import sys
import argparse
from network_features import get_undirected_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Extract data',
        description = 'Construct dataset for supervised link prediction from the CondMat network dataset')

    parser.add_argument('n', type=int, help='Degree of the neighborhood to extract', choices=[2, 3, 4])
    args = parser.parse_args()

    n_deg = args.n
        
    edges_dict = {}
    with open('data/ca-CondMat.txt', "r") as f:
        lines = f.readlines()
        for line in lines[4:]:
            data = line.strip().split("\t")
            if (int(data[0]), int(data[1])) in edges_dict:
                edges_dict[(int(data[0]), int(data[1]))] += 1
            elif (int(data[1]), int(data[0])) in edges_dict:
                edges_dict[(int(data[1]), int(data[0]))] += 1
            else:
                edges_dict[(int(data[0]), int(data[1]))] = 1
            
    edges = [(k[0], k[1], v) for k, v in edges_dict.items()]

    feature_edges = random.sample(edges, int(len(edges)*0.7))
    label_edges = {(e[0], e[1]): 1 for e in list(set(edges) - set(feature_edges))}

    features = get_undirected_features(feature_edges, label_edges, n_deg, True)

    print("Writing to file...")
    features.to_csv(f"data/clean_datasets/condmat_{n_deg}.csv")

    print("Done!")