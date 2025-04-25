'''
Homework 5
Samarth Kumar
'''

import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict, Any

# Reading the .pir file
def read_file(path = 'msa.pir'):
    names = []
    sequences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                names.append(line[1:].strip())
                sequences.append('')
            else:
                sequences[-1] += line
    return names, sequences

# Hamming distance -> Number of differences between 2 sequences
def hamming_distance(sequence1, sequence2):
    # Lengths must be equal
    if len(sequence1) != len(sequence2):
        raise ValueError('Sequences must be the same length')
    count = 0
    for ch1, ch2 in zip(sequence1, sequence2):
        if ch1 != ch2:
            count += 1
    return count

# Generating the original distance matrix between the sequences, using hamming distance
def original_matrix(names, seqs):
    n = len(names)
    matrix = pd.DataFrame(np.zeros((n, n), dtype=int), index=names, columns=names)
    for i in range(n):
        for j in range(n):
            matrix.iat[i, j] = hamming_distance(seqs[i], seqs[j])
    return matrix

# UPGMA clustering -> getting the edges and creating internal nodes that'll be used to
# get the adjacency list and construct a new distance matrix with the added nodes.
def upgma(distance_matrix):
    # Initialize clusters from original sequences.
    clusters: Dict[str, Dict[str, Any]] = {
        name: {'height': 0.0, 'size': 1} for name in distance_matrix.index
    }
    dists = distance_matrix.astype(float).copy()
    edges: List[Tuple[str, str, float]] = []
    count = 1

    while len(clusters) > 1:
        # Finding the closest pair
        a, b, best = None, None, float('inf')
        for i in dists.index:
            for j in dists.columns:
                if i == j:
                    continue
                if dists.at[i, j] < best:
                    a, b, best = i, j, dists.at[i, j]

        # create new internal node
        new_node = f"Node {count}"
        count += 1
        new_height = best / 2.0

        # Computing the branch lengths
        length_a = new_height - clusters[a]['height']
        length_b = new_height - clusters[b]['height']
        edges.append((new_node, a, length_a))
        edges.append((new_node, b, length_b))

        # Updating the clusters
        size_a = clusters[a]['size']
        size_b = clusters[b]['size']
        clusters[new_node] = {
            'height': new_height,
            'size': size_a + size_b
        }
        del clusters[a], clusters[b]

        # Getting the distances from new node to the other nodes
        new_row: Dict[str, float] = {}
        for other in list(dists.index):
            if other in (a, b):
                continue
            d1 = dists.at[other, a]
            d2 = dists.at[other, b]

            # Distance formula
            new_row[other] = (size_a * d1 + size_b * d2) / (size_a + size_b)

        # Drop old rows/cols, add the new one
        dists = dists.drop(index=[a, b], columns=[a, b])
        dists.loc[new_node, new_node] = 0.0
        for other, val in new_row.items():
            dists.at[new_node, other] = val
            dists.at[other, new_node] = val

    return edges

# Building adjacency list from the edges determined by UPGMA clustering.
def get_adjacency(edges):
    adjacency_list = {}
    for u, v, w in edges:
        adjacency_list.setdefault(u, []).append((v, w))
        adjacency_list.setdefault(v, []).append((u, w))
    return adjacency_list

# BFS algorithm -> helper function that uses adjacency list 
# and computes distance between 2 nodes.
def bfs(adjacency_list, source, target):
    visited = {source}
    queue = deque([(source, 0.0)])
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for neighbor, weight in adjacency_list.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + weight))
    return float('NaN')

# Using BFS to get the distance matrix with the added internal nodes.
def internal_matrix(adjacency_list):
    nodes = sorted(adjacency_list.keys())
    mat = pd.DataFrame(index=nodes, columns=nodes, dtype=float)
    for a in nodes:
        for b in nodes:
            mat.at[a, b] = bfs(adjacency_list, a, b)
    return mat

# Creating the DOT file to actually visualize the phylogenetic tree.
def create_DOT(names, edges, leaf_color = 'lightgreen'):
    lines = [
        'graph phylo_tree {',
        '    node [shape=circle];'
    ]
    
    # Applying color to the main sequence nodes and labeling the weighted edges
    for leaf in names:
        lines.append(f'    "{leaf}" [style=filled, fillcolor={leaf_color}];')
    for u, v, w in edges:
        lines.append(f'    "{u}" -- "{v}" [label={w}, weight={w}];')
    lines.append('}')
    return '\n'.join(lines)

# Saving the distance matrices to a file.
def save_matrix_file(df, file_name):
    Path(file_name).write_text(df.to_string())
    print(f'Saved to {file_name}')

def main():
    names, sequences = read_file()

    # Original distance matrix
    distance_matrix = original_matrix(names, sequences)
    print(f'Original distance matrix:\n{distance_matrix}')
    save_matrix_file(distance_matrix, 'distance_matrix_original.txt')
    
    # Distance matrix with the added internal nodes.
    edges = upgma(distance_matrix)
    adjacency_list = get_adjacency(edges)
    final_matrix = internal_matrix(adjacency_list)
    print(f'\nDistance Matrix (with Internal Nodes):\n{final_matrix}')
    save_matrix_file(final_matrix, 'distance_matrix_internal.txt')
    
    # Phylogenetic tree, DOT file format
    dot_str = create_DOT(names, edges)
    Path('phylogenetic_tree.dot').write_text(dot_str)
    print(f'\nPhylogenetic Tree (DOT):\n{dot_str}')
    print("Saved to phylogenetic_tree.dot")

if __name__ == '__main__':
    main()
