import argparse
import pandas as pd
import matplotlib.pyplot as plt
from os import path

import tabulate

from wikisearch.consts.mongo import CSV_SEPARATOR


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
    parser.add_argument('-s', '--statistics', help='Path to the statistics file')
    parser.add_argument('-c', '--compared_methods', help='The methods to compare between their distances results',
                        nargs=2)
    args = parser.parse_args()

    statistics = pd.read_csv(args.statistics, sep=CSV_SEPARATOR)
    distance_type_1 = args.compared_methods[0]
    distance_type_2 = args.compared_methods[1]
    first_distances = statistics[distance_type_1]
    second_distances = statistics[distance_type_2]

    output_dir = path.dirname(args.statistics)

    # Generate differences histogram
    differences = (first_distances - second_distances).round().astype(int)
    differences_ticks = range((min(differences)), max(differences) + 2)
    create_histogram(differences, differences_ticks,
                     f"Differences between {distance_type_1} to {distance_type_2}",
                     output_dir, f"{distance_type_1}_{distance_type_2}_differences_histogram.jpg")

    # Generate absolute differences histogram
    abs_differences = abs(first_distances - second_distances).round().astype(int)
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
    print(tabulate.tabulate(statistics_df, headers='keys', tablefmt='grid', floatfmt='.2f', showindex=False))
    with open(path.join(output_dir, "distances_differences.stats"), 'w', encoding='utf8') as f:
        f.write(tabulate.tabulate(statistics_df, headers='keys', showindex=False, tablefmt='grid', floatfmt='.2f'))
