import argparse
import os
import time
from collections import Counter
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import torch.utils.data

from scripts.loaders import load_embedder_from_model_path, load_model_from_path
from scripts.utils import print_progress_bar
from wikisearch.astar import Astar
from wikisearch.consts.mongo import CSV_SEPARATOR
from wikisearch.consts.statistics_column_names import *
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy


def create_histogram(values, values_ticks, title, output_path, histogram_name):
    plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.xlabel("Differences")
    plt.ylabel("# Occurences")
    counts, _, _ = plt.hist(values, bins=values_ticks, align='left')
    plt.gca().set_xticks(values_ticks[:-1])
    for i, distance in enumerate(values_ticks[:-1]):
        plt.text(distance, counts[i] + 1, str(int(counts[i])))
    plt.savefig(path.join(output_path, histogram_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file. When running from linux - '
                                              'notice to not put a \'/\' after the file name')
    parser.add_argument('-df', '--dataset_file', help='Path to a dataset file')
    args = parser.parse_args()

    output_dir = path.dirname(args.model)
    model_file_name = path.splitext(path.basename(args.model))[0]

    embedder = load_embedder_from_model_path(args.model)
    model = load_model_from_path(args.model)

    # Loads the dataset file
    dataset = pd.read_csv(args.dataset_file, sep=CSV_SEPARATOR).values

    # Prepare the statistics table
    pd.set_option('display.max_columns', 10)
    pd.set_option('precision', 2)
    statistics_df = pd.DataFrame(columns=[SRC_NODE, DST_NODE, BFS_DIST, NN_DIST])
    cost = UniformCost(1)
    heuristic = NNHeuristic(model, embedder)
    strategy = DefaultAstarStrategy()
    graph = WikiGraph()
    astar = Astar(cost, heuristic, strategy, graph)
    with torch.no_grad():
        start = time.time()
        for idx, (source, destination, actual_distance) in enumerate(dataset, 1):
            # _, astar_distance, _ = astar.run(tokenize_title(source), tokenize_title(destination))
            statistics_df = statistics_df.append(
                {
                    SRC_NODE: source,
                    DST_NODE: destination,
                    BFS_DIST: actual_distance,
                    NN_DIST: model(embedder.embed(source).unsqueeze(0), embedder.embed(destination).unsqueeze(0)).round().int().item()
                    # ASTAR_DIST: astar_distance
                }, ignore_index=True)
            print_progress_bar(idx, len(dataset), time.time() - start, prefix=f'Progress: ', length=50)

    # Print out the statistics to csv file
    statistics_file_path = path.join(output_dir, f"{model_file_name}.stats")
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    statistics_df.to_csv(statistics_file_path, sep=CSV_SEPARATOR, header=True, index=False)

    # Compare between BFS to NN
    # TODO: Try to work directly with statistics_df and not reading the csv as it quite stupid
    statistics = pd.read_csv(statistics_file_path, sep=CSV_SEPARATOR)
    distance_type_1 = BFS_DIST
    distance_type_2 = NN_DIST
    first_distances = statistics[distance_type_1]
    second_distances = statistics[distance_type_2]

    # Generate distances histogram
    plt.figure(figsize=(16, 9))
    plt.title(f"Distances {distance_type_1} vs. {distance_type_2}")
    plt.xlabel("Distances")
    plt.ylabel("# Occurences")
    width = 0.3
    first_distances_ticks = np.arange(min(first_distances), max(first_distances) + 1)
    first_distances_counter = Counter(first_distances)
    plt.bar(first_distances_ticks - width / 2,
            [first_distances_counter[distance] for distance in first_distances_ticks], width=width, align='center')
    second_distances_ticks = np.arange(min(second_distances), max(second_distances) + 1)
    second_distances_counter = Counter(second_distances)
    plt.bar(second_distances_ticks + width / 2,
            [second_distances_counter[distance] for distance in second_distances_ticks], width=width, align='center')
    plt.legend([distance_type_1, distance_type_2])
    plt.savefig(path.join(os.path.join(output_dir, f"{distance_type_1}_{distance_type_2}_distances_histogram.jpg")))

    # Generate differences histogram
    differences = first_distances - second_distances
    differences_ticks = range((min(differences)), max(differences) + 2)
    create_histogram(differences, differences_ticks,
                     f"Differences between {distance_type_1} to {distance_type_2}",
                     output_dir, f"{distance_type_1}_{distance_type_2}_differences_histogram.jpg")

    # Generate absolute differences histogram
    abs_differences = abs(first_distances - second_distances)
    abs_differences_ticks = range(min(abs_differences), max(abs_differences) + 2)
    create_histogram(abs_differences, abs_differences_ticks,
                     f"Absolute Differences between {distance_type_1} to {distance_type_2}",
                     output_dir, f"{distance_type_1}_{distance_type_2}_abs_differences_histogram.jpg")

    # Options used for printing dataset summaries and statistics
    pd.set_option('display.max_columns', 10)
    pd.set_option('precision', 2)
    statistics_df = pd.DataFrame(columns=['Methods Compared', 'Average Difference', 'Std', 'Average Abs Difference',
                                          'Std for Abs', '50% Percentage'])
    sorted_differences = abs_differences.sort_values().get_values()
    differences_length = len(abs_differences)
    statistics_df = statistics_df.append(
        {
            'Methods Compared': f"{distance_type_1} to {distance_type_2}",
            'Average Difference': differences.mean(),
            'Std': differences.std(),
            'Average Abs Difference': abs_differences.mean(),
            'Std for Abs': abs_differences.std(),
            '50% Percentage': abs_differences.median(),
            '75% Percentage': sorted_differences[round(0.75*differences_length)],
            '90% Percentage': sorted_differences[round(0.90 * differences_length)]
        }, ignore_index=True)

    # Print out statistics to file
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    print(tabulate.tabulate(statistics_df, headers='keys', tablefmt='fancy_grid', floatfmt='.2f', showindex=False))
    with open(path.join(output_dir, "distances_differences.stats"), 'w', encoding='utf8') as f:
        f.write(tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='fancy_grid', floatfmt='.2f'))
