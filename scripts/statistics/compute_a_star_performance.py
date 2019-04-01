import argparse
import time
from collections import defaultdict
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import torch.utils.data

from scripts.consts.statistics import *
from scripts.loaders import load_embedder_from_model_path, load_model_from_path, load_embedder_by_name, \
    load_distance_method
from scripts.utils import print_progress_bar, timing
from wikisearch.astar import Astar
from wikisearch.consts.mongo import CSV_SEPARATOR
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.graph import WikiGraph
from wikisearch.heuristics import BFSHeuristic
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy
from wikisearch.utils.clean_data import tokenize_title


def print_path(path_pr):
    return "->".join([f"'{node.title}'" for node in path_pr])


def calculate_averages(distances_times: dict):
    distances = []
    average_values = []
    std = []
    for distance, distance_times in distances_times.items():
        distance_times = np.array(distance_times)
        average_time = distance_times.mean()

        distances.append(distance)
        average_values.append(average_time)
        std.append(distance_times.std())

    return np.array(distances), np.array(average_values), np.array(std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file. When running from linux - '
                                              'notice to not put a \'/\' after the file name')
    parser.add_argument('-df', '--dataset_file', help='Path to a dataset file')
    parser.add_argument('-c', '--cost', default=1, help='The cost for the customizable model')
    parser.add_argument('-e', '--embedding_name', help='The embedder name')
    parser.add_argument('-hd', '--heuristic_distance', help='The heuristic distance method')
    args = parser.parse_args()

    # embedder = load_embedder_from_model_path(args.model)
    # model = load_model_from_path(args.model)
    embedder_by_name = load_embedder_by_name(args.embedding_name)
    heuristic_distance_method = load_distance_method(args.heuristic_distance, embedder_by_name)

    # Loads the dataset file
    dataset = pd.read_csv(args.dataset_file, sep=CSV_SEPARATOR).values

    # Prepare the statistics table
    statistics_df = pd.DataFrame(columns=[SRC_NODE, DST_NODE,
                                          BFS_DIST, BFS_TIME, BFS_DEVELOPED, BFS_H_DEVELOPED, BFS_PATH,
                                          NN_DIST, NN_TIME, NN_DEVELOPED, NN_H_DEVELOPED, NN_PATH])
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')

    strategy = DefaultAstarStrategy()
    graph = WikiGraph()

    astar_bfs = Astar(UniformCost(1), BFSHeuristic(), strategy, graph)
    # astar_nn = Astar(cost, NNHeuristic(model, embedder), strategy, graph)
    astar_nn = Astar(UniformCost(int(args.cost)), heuristic_distance_method, strategy, graph)

    dataset_len = len(dataset)
    bfs_distance_times = defaultdict(list)
    bfs_distance_developed = defaultdict(list)
    nn_distance_times = defaultdict(list)
    nn_distance_developed = defaultdict(list)
    nn_computed_distance = defaultdict(list)

    # Parameters to save the result to a file
    model_dir_path = path.dirname(args.model)
    model_file_name = path.splitext(path.basename(args.model))[0]
    statistics_file_path = path.join(model_dir_path, f"{model_file_name}_{args.embedding_name}_{args.heuristic_distance}.a_star_stats")
    with torch.no_grad():
        start = time.time()
        for idx, (source, destination, _) in enumerate(dataset, 1):
            tokenized_source = tokenize_title(source)
            tokenized_destination = tokenize_title(destination)
            bfs_path, bfs_dist, bfs_developed, bfs_time = \
                timing(astar_bfs.run, tokenized_source, tokenized_destination)
            nn_path, nn_dist, nn_developed, nn_time = \
                timing(astar_nn.run, tokenized_source, tokenized_destination)
            bfs_distance_times[bfs_dist].append(bfs_time)
            bfs_distance_developed[bfs_dist].append(bfs_developed)
            nn_distance_times[bfs_dist].append(nn_time)
            nn_distance_developed[bfs_dist].append(nn_developed)
            nn_computed_distance[bfs_dist].append(nn_dist)
            statistics_df = statistics_df.append(
                {
                    SRC_NODE: source,
                    DST_NODE: destination,
                    BFS_DIST: bfs_dist,
                    BFS_TIME: bfs_time,
                    BFS_DEVELOPED: bfs_developed,
                    BFS_H_DEVELOPED: astar_bfs._heuristic.count,
                    BFS_PATH: print_path(bfs_path).replace("->", "\n->"),
                    NN_DIST: nn_dist,
                    NN_TIME: nn_time,
                    NN_DEVELOPED: nn_developed,
                    NN_H_DEVELOPED: astar_nn._heuristic.count,
                    NN_PATH: print_path(nn_path).replace("->", "\n->")
                }, ignore_index=True)
            # Print out the statistics as tabulate
            statistics_df_tabulate = \
                tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='fancy_grid',
                                  floatfmt='.5f')
            with open(statistics_file_path, 'w', encoding='utf8') as f:
                f.write(statistics_df_tabulate)
            print_progress_bar(idx, dataset_len, time.time() - start, prefix=f'A*', length=50)

    # Creates the distance-time statistics
    width = 0.3

    bfs_distances_time, bfs_average_times, bfs_std_time = \
        calculate_averages(bfs_distance_times)
    nn_distances_time, nn_average_times, nn_std_time = \
        calculate_averages(nn_distance_times)

    plt.title("A* running times")
    plt.xlabel("Distance")
    plt.ylabel("Time")
    plt.bar(bfs_distances_time - width / 2, bfs_average_times, yerr=bfs_std_time, width=width, align='center')
    plt.bar(nn_distances_time + width / 2, nn_average_times, yerr=nn_std_time, width=width, align='center')
    plt.legend([BFS_TIME, NN_TIME])
    plt.savefig(path.join(model_dir_path, f"a_star_running_time_{args.embedding_name}_{args.heuristic_distance}.jpg"))

    plt.figure()

    # Creates the distance-#developed statistics
    bfs_distances_developed, bfs_average_developed, bfs_std_developed = \
        calculate_averages(bfs_distance_developed)
    nn_distances_developed, nn_average_developed, nn_std_developed = \
        calculate_averages(nn_distance_developed)

    plt.title("A* developed")
    plt.xlabel("Distance")
    plt.ylabel("#Developed")
    plt.bar(bfs_distances_developed - width / 2, bfs_average_developed,
            yerr=bfs_std_developed, width=width, align='center')
    plt.bar(nn_distances_developed + width / 2, nn_average_developed,
            yerr=nn_std_developed, width=width, align='center')
    plt.legend([BFS_DEVELOPED, NN_DEVELOPED])
    plt.savefig(path.join(model_dir_path, f"a_star_#developed_{args.embedding_name}_{args.heuristic_distance}.jpg"))

    # Creates the actual_distance-computed_distance statistics
    actual_distances, nn_average_computed_distances, nn_std_computed_distances = \
        calculate_averages(nn_computed_distance)

    plt.title("Customizable model distances")
    plt.xlabel("Actual Distance")
    plt.ylabel("Computed Distance")
    plt.bar(actual_distances, nn_average_computed_distances, yerr=nn_std_computed_distances, align='center')
    for actual_distance, average_distance in zip(actual_distances, nn_average_computed_distances):
        plt.text(actual_distance - 0.1, average_distance, str(average_distance))
    plt.legend([NN_DEVELOPED])
    plt.savefig(path.join(model_dir_path, f"a_star_#developed_{args.embedding_name}_{args.heuristic_distance}.jpg"))
