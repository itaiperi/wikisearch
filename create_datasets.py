import argparse
import os
import random

import pandas as pd

from wikisearch.astar import Astar
from wikisearch.graph import WikiGraph
from wikisearch.heuristics.bfs_heuristic import bfs_heuristic
from wikisearch.strategies.default_astar_strategy import DefaultAstarStrategy

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_records', help="Number of records for training, validation, test sets", nargs=3,
                    type=int, required=True)
parser.add_argument('-s', '--seed', help="Random seed", type=int)
parser.add_argument('-o', '--out', help="Output dir path", required=True)

args = parser.parse_args()
rnd_generator = random.Random(args.seed)  # If args.seed is None, system's time is used (default behavior)

heuristic = bfs_heuristic
strategy = DefaultAstarStrategy()
graph = WikiGraph()
astar = Astar(heuristic, strategy, graph)

graph_keys = sorted(graph.keys())

for i, dataset_type in enumerate(['train', 'validation', 'test']):
    dataset = []
    for i in range(args.num_records[i]):  # Training set
        source, dest = rnd_generator.sample(graph_keys, 2)
        _, distance, _ = astar.run(source, dest)
        dataset.append((source, dest, distance))

    # Create dataframe from dataset
    df = pd.DataFrame.from_records(dataset, columns=['source', 'destination', 'min_distance'])
    # Define path to save dataset to
    dataset_path = os.path.abspath(os.path.join(args.out, dataset_type + '.csv'))
    # Save dataset (through dataframe)
    df.to_csv(dataset_path, header=True, index=False)

print("Dataset created at", os.path.abspath(args.out))
