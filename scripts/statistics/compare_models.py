import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate

from scripts.consts.model import MODEL_META
from scripts.consts.statistics import NN_DIST
from wikisearch.consts.mongo import CSV_SEPARATOR

# Prepare the statistics table
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_colwidth', 30)
pd.set_option('precision', 2)


def generate_models_results(model_dir_paths, gt_results_path):
    models_df = pd.read_csv(gt_results_path, sep=CSV_SEPARATOR)

    for model_index, (model_dir_path, model_dir_name) in enumerate(zip(model_dir_paths, model_dir_names), 1):
        with open(os.path.join(model_dir_path, MODEL_META)) as meta_file:
            model_meta = json.load(meta_file)
        model_metas.append(model_meta)
        model_df = pd.read_csv(os.path.join(model_dir_path, 'model.stats'), sep=CSV_SEPARATOR)
        models_df[model_dir_name] = model_df[NN_DIST]

    return models_df


def get_statistics(models_df, model_names):
    statistics_df = pd.DataFrame(columns=['Model Name', 'Admissableness', 'Average Difference', 'Std',
                                          'Average Abs Difference', 'Std for Abs', '50% Percentage', '75% Percentage',
                                          '90% Percentage'])

    for model_name in model_names:
        differences = models_df['min_distance'] - models_df[model_name]
        abs_differences = abs(differences)
        differences_length = len(abs_differences)
        statistics_df = statistics_df.append(
            {
                'Model Name': model_name,
                'Admissableness': f"{sum(differences >= 0) / differences_length * 100:.1f}%",
                'Average Difference': differences.mean(),
                'Std': differences.std(),
                'Average Abs Difference': abs_differences.mean(),
                'Std for Abs': abs_differences.std(),
                '50% Percentage': abs_differences.median(),
                '75% Percentage': abs_differences.quantile(0.75),
                '90% Percentage': abs_differences.quantile(0.9),
            }, ignore_index=True)
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    return statistics_df


def create_histogram(values, values_ticks, title, output_path, histogram_name):
    plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.xlabel("Differences")
    plt.ylabel("# Occurences")
    counts, _, _ = plt.hist(values, bins=values_ticks, align='left')
    plt.gca().set_xticks(values_ticks[:-1])
    for i, distance in enumerate(values_ticks[:-1]):
        plt.text(distance, counts[i] + 1, str(int(counts[i])))
    plt.savefig(os.path.join(output_path, histogram_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', required=True, nargs='+',
                        help='Path to the models\' directories.')
    parser.add_argument('-df', '--dataset_file', required=True, help='Path to a dataset file')
    parser.add_argument('-o', '--out', required=True, help="Path to output directory")
    args = parser.parse_args()

    os.makedirs(args.out)
    model_dir_names = [os.path.basename(os.path.normpath(model_dir_path)) for model_dir_path in args.models]
    model_metas = []

    models_df = generate_models_results(args.models, args.dataset_file)
    # Print out the distances to csv file
    distances_file_path = os.path.join(args.out, f"distances.stats")
    models_df.to_csv(distances_file_path, sep=CSV_SEPARATOR, header=True, index=False)

    statistics_df = get_statistics(models_df, model_dir_names)
    # Print out statistics to file
    print(tabulate.tabulate(statistics_df, headers='keys', tablefmt='fancy_grid', floatfmt='.2f', showindex=False))
    with open(os.path.join(args.out, "distances_differences.stats"), 'w', encoding='utf8') as f:
        f.write(
            tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='fancy_grid', floatfmt='.2f'))

    # Add 1 for ground truth, and 1 for spacing
    bar_width = 1 / (len(args.models) + 2)
    # Generate distances histogram
    plt.figure(figsize=(16, 9))
    plt.title(f"Distances Ground Truth vs. Various Models")
    plt.xlabel("Distances")
    plt.ylabel("# Occurrences")
    # Add 1 for ground truth distance
    models_parity = (len(model_dir_names) + 1) % 2
    # Add 1 for ground truth
    for index, model in enumerate(model_dir_names + ['min_distance'], -((len(model_dir_names) + 1) // 2)):
        ticks = np.arange(min(models_df[model]), max(models_df[model]) + 1)
        counter = Counter(models_df[model])
        plt.bar(ticks + (index + (0 if models_parity else 0.5)) * bar_width, [counter[distance] for distance in ticks], width=bar_width, align='center')
    plt.legend(model_dir_names + ['Ground Truth'])
    plt.savefig(os.path.join(os.path.join(args.out, "distances_histogram.jpg")))

    # Generate differences histogram
    plt.figure(figsize=(16, 9))
    plt.title(f"Differences from Ground Truth for Various Models")
    plt.xlabel("Differences")
    plt.ylabel("# Occurrences")
    models_parity = len(model_dir_names) % 2
    for index, model in enumerate(model_dir_names, -(len(model_dir_names) // 2)):
        differences = models_df['min_distance'] - models_df[model]
        ticks = np.arange(min(differences), max(differences) + 1)
        counter = Counter(differences)
        plt.bar(ticks + (index + (0 if models_parity else 0.5)) * bar_width, [counter[distance] for distance in ticks], width=bar_width, align='center')
    plt.legend(model_dir_names)
    plt.savefig(os.path.join(os.path.join(args.out, "differences_histogram.jpg")))

    # Generate absolute differences histogram
    plt.figure(figsize=(16, 9))
    plt.title(f"Absolute Differences from Ground Truth for Various Models")
    plt.xlabel("Differences")
    plt.ylabel("# Occurrences")
    models_parity = len(model_dir_names) % 2
    for index, model in enumerate(model_dir_names, -(len(model_dir_names) // 2)):
        abs_differences = abs(models_df['min_distance'] - models_df[model])
        ticks = np.arange(min(abs_differences), max(abs_differences) + 1)
        counter = Counter(abs_differences)
        plt.bar(ticks + (index + (0 if models_parity else 0.5)) * bar_width, [counter[distance] for distance in ticks],
                width=bar_width, align='center')
    plt.legend(model_dir_names)
    plt.savefig(os.path.join(os.path.join(args.out, "abs_differences_histogram.jpg")))
