import argparse
import os
import random

import pandas as pd

from wikisearch.consts.mongo import WIKI_LANG
from wikisearch.graph import WikiGraph

rnd_generator = random.Random()


def find_at_distance(graph, source_node, desired_distance):
    if not len(list(source_node.get_neighbors())):
        return None, 0

    actual_distance = 0
    current_distance_nodes = {source_node}
    all_nodes = set(current_distance_nodes)

    while actual_distance < desired_distance:
        # If neighbor has been found before, then there's a shorter path, and we don't add it to current distance
        next_distance_nodes = {neighbor for node in current_distance_nodes
                               for neighbor in graph.get_node_neighbors(node) if neighbor not in all_nodes}
        all_nodes.update(next_distance_nodes)
        if next_distance_nodes:
            actual_distance += 1
            current_distance_nodes = next_distance_nodes
        else:
            # No neighboring nodes (in shortest distance), so we break and choose one at random
            break

    if not actual_distance:
        return None, actual_distance
    return rnd_generator.choice(list(current_distance_nodes)), actual_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_records', help="Number of records for training, validation, test sets", nargs=3,
                        type=int, required=True)
    parser.add_argument('-s', '--seed', help="Random seed", type=int)
    parser.add_argument('-o', '--out', help="Output dir path", required=True)
    parser.add_argument('-d', '--max_distance', type=int, default=20)
    args = parser.parse_args()

    if args.max_distance < 1:
        raise ValueError("Distance is not a positive integer")

    rnd_generator.seed(args.seed)  # If args.seed is None, system's time is used (default behavior)

    graph = WikiGraph(WIKI_LANG)
    graph_keys = sorted(graph.keys())

    for dataset_type, num_records in zip(['train', 'validation', 'test'], args.num_records):
        dataset = []
        for i in range(num_records):
            dest = None
            source = ""
            desired_distance = rnd_generator.randint(1, args.max_distance)
            distance = 0
            while dest is None:  # This is to make sure that the source node actually has neighbors in the first place
                source = rnd_generator.choice(graph_keys)
                dest, distance = find_at_distance(graph, graph.get_node(source), desired_distance)
            # print("%s\t%s\t%d\t%d" % (source, dest.title, distance, desired_distance))
            dataset.append((source, dest.title, distance))

        # Create dataframe from dataset
        df = pd.DataFrame.from_records(dataset, columns=['source', 'destination', 'min_distance'])
        # Define path to save dataset to
        dataset_path = os.path.abspath(os.path.join(args.out, dataset_type + '.csv'))
        # Save dataset (through dataframe)
        df.to_csv(dataset_path, header=True, index=False, sep='\t')

    print("Dataset created at", os.path.abspath(args.out))
