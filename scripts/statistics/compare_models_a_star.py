import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.consts.model import MODEL_ASTAR_CSV
from scripts.consts.statistics import NN_DIST, NN_DEVELOPED, NN_TIME, BFS_DIST, BFS_TIME, BFS_DEVELOPED
from wikisearch.consts.mongo import CSV_SEPARATOR

# Prepare the statistics table
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 30)
pd.set_option('precision', 2)

# Constant strings
BFS = "BFS"
DEVELOPED = "Developed"
TIME = "Time"
DISTANCE = "Distance"


def generate_models_results(model_dir_paths):
    models_df_local = pd.DataFrame()

    # TODO: after running the models again on astar, uncomment the separator
    model_df = pd.read_csv(os.path.join(model_dir_paths[0], MODEL_ASTAR_CSV))#, sep=CSV_SEPARATOR)
    models_df_local[f"{BFS}_{DEVELOPED}"] = model_df[BFS_DEVELOPED]
    models_df_local[f"{BFS}_{TIME}"] = model_df[BFS_TIME]
    models_df_local[f"{BFS}_{DISTANCE}"] = model_df[BFS_DIST]

    for model_dir_path, model_dir_name in zip(model_dir_paths, model_dir_names):
        # TODO: after running the models again on astar, uncomment the separator
        model_df = pd.read_csv(os.path.join(model_dir_path, MODEL_ASTAR_CSV))#, sep=CSV_SEPARATOR)
        models_df_local[f"{model_dir_name}_{DEVELOPED}"] = model_df[NN_DEVELOPED]
        models_df_local[f"{model_dir_name}_{TIME}"] = model_df[NN_TIME]
        models_df_local[f"{model_dir_name}_{DISTANCE}"] = model_df[NN_DIST]

    return models_df_local


def generate_histogram(title):
    plt.figure(figsize=(16, 9))
    plt.title(f"{title} BFS vs. Various Models")
    plt.xlabel("Distance")
    plt.ylabel(title)
    # Add 1 for ground truth distance
    models_parity = (len(model_dir_names) + 1) % 2
    # Add 1 for ground truth
    ticks = np.arange(min(models_df[f"{BFS}_{DISTANCE}"]), max(models_df[f"{BFS}_{DISTANCE}"]) + 1)
    aggregated_models_by_distance = models_df.groupby(BFS_DIST)
    for index, model in enumerate(model_dir_names, -((len(model_dir_names) + 1) // 2)):
        cur_model_df = aggregated_models_by_distance[f"{model}_{title}"]
        plt.bar(ticks + (index + (0 if models_parity else 0.5)) * bar_width,
                cur_model_df.mean(), yerr=cur_model_df.std(), width=bar_width, align='center')
    plt.legend(model_dir_names)
    plt.savefig(os.path.join(os.path.join(models_base_dir, f"{title.lower()}_histogram.jpg")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', required=True, nargs='+', help='Path to the models\' directories.')
    args = parser.parse_args()

    model_dir_names = [os.path.basename(os.path.normpath(model_dir_path)) for model_dir_path in args.models]
    models_base_dir = os.path.dirname(args.models[0])

    models_df = generate_models_results(args.models)
    model_dir_names.append("BFS")

    # Add 1 for BFS, and 1 for spacing
    bar_width = 1 / (len(args.models) + 2)

    generate_histogram(DEVELOPED)
    generate_histogram(TIME)
    generate_histogram(DISTANCE)
