import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tabulate

from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG
from wikisearch.graph import WikiGraph

# Options used for printing dataset summaries and statistics
pd.set_option('display.max_columns', 10)
pd.set_option('precision', 2)

dataset_types = ['train', 'validation', 'test']
rnd_generator = random.Random()


def find_at_distance(graph, source_node, desired_distance):
    """
    Find a node at desired distance from source node
    :param graph: wikisearch.WikiGraph instance
    :param source_node: wikisearch.GraphNode instance of source page
    :param desired_distance: distance (minimal) at which a node should be found
    :return: node at desired distance / shorter, if there are no nodes at such distance, and the real distance
    """
    if not list(source_node.get_neighbors()):
        return None, 0

    actual_distance = 0
    current_distance_nodes = {source_node}
    all_nodes = set(current_distance_nodes)

    while actual_distance < desired_distance:
        # If neighbor has been found before, then there's a shorter path, and we don't add it to current distance
        next_distance_nodes = {neighbor for node in current_distance_nodes
                               for neighbor in graph.get_node_neighbors(node)}
        next_distance_nodes = next_distance_nodes - all_nodes
        all_nodes.update(next_distance_nodes)
        if next_distance_nodes:
            actual_distance += 1
            current_distance_nodes = next_distance_nodes
        else:
            # No neighboring nodes (in shortest distance), so we break and choose one at random
            break

    if not actual_distance:
        # Edge case, where there are no neighbors
        return None, actual_distance
    # Return a random neighbor at actual_distance away from source page
    return rnd_generator.choice(list(current_distance_nodes)), actual_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_records', '-n', help='Number of records for training, validation, test sets', nargs=3,
                        type=int, required=True)
    parser.add_argument('--seed', '-s', type=int, help='Seed used by random generator')
    parser.add_argument('--out', '-o', required=True, help='Output dir path')
    parser.add_argument('--max_distance', '-d', type=int, default=20, help='Maximum distance to search for')
    args = parser.parse_args()

    if args.max_distance < 1:
        raise ValueError('Distance is not a positive integer')

    rnd_generator.seed(args.seed)  # If args.seed is None, system's time is used (default behavior)

    graph = WikiGraph(WIKI_LANG)
    graph_keys = sorted(graph.keys())

    entire_start = time.time()
    distances = defaultdict(list)
    runtimes = {}
    # Go over all types of datasets
    for dataset_type, num_records in zip(dataset_types, args.num_records):
        dataset_start = time.time()
        dataset = []
        # Build current dataset
        for i in range(num_records):
            dest = None
            source = None
            desired_distance = rnd_generator.randint(1, args.max_distance)
            distance = 0
            while dest is None:  # This is to make sure that the source node actually has neighbors in the first place
                source = rnd_generator.choice(graph_keys)
                dest, distance = find_at_distance(graph, graph.get_node(source), desired_distance)
            distances[dataset_type].append(distance)
            dataset.append((source, dest.title, distance))
            print_progress_bar(i + 1, num_records, time.time() - dataset_start, prefix=dataset_type.capitalize(), length=50)
        print(f'{dataset_type.capitalize()}: {num_records} datapoints created.')

        # Create dataframe from dataset
        df = pd.DataFrame.from_records(dataset, columns=['source', 'destination', 'min_distance'])
        # Define path to save dataset to
        dataset_path = os.path.abspath(os.path.join(args.out, '_'.join([dataset_type, str(args.seed), str(num_records)]) + '.csv'))
        # Save dataset (through dataframe)
        df.to_csv(dataset_path, header=True, index=False, sep='\t')
        runtimes[dataset_type] = time.time() - dataset_start

    # Create statistics for dataset
    statistics_df = pd.DataFrame(columns=['Dataset', 'Number of entries', 'Build time', 'Average build time/entry',
                                          'Min distance', 'Max distance', 'Average distance', 'Standard deviation'])
    for dataset_type, num_records in zip(dataset_types, args.num_records):
        statistics_df = statistics_df.append({'Dataset': dataset_type.capitalize(),
                                              'Number of entries': num_records,
                                              'Build time': runtimes[dataset_type],
                                              'Average build time/entry': runtimes[dataset_type] / num_records,
                                              'Min distance': np.min(distances[dataset_type]),
                                              'Max distance': np.max(distances[dataset_type]),
                                              'Average distance': np.mean(distances[dataset_type]),
                                              'Standard deviation': np.std(distances[dataset_type])},
                                             ignore_index=True)

    # Print out statistics to file
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    print(tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='grid', floatfmt='.2f'), )
    with open(os.path.join(args.out, 'stats.txt'), 'w') as f:
        f.write(tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='grid', floatfmt='.2f'))
    print('Total elapsed time for all datsets: {:.1f} seconds.'.format(time.time() - entire_start))
