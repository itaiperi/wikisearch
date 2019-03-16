import argparse
from os import path
import time
import random

import pandas as pd

from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import CSV_SEPARATOR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Path to testing file")
    parser.add_argument("-max", "--max_distance", default=14, help="The maximum distance exist in the dataset")
    parser.add_argument("-a", "--amount_per_distance", default=10, help="amount of couples per distance")

    args = parser.parse_args()

    test_dataset = pd.read_csv(args.test, sep=CSV_SEPARATOR).values
    dataset_len = len(test_dataset)

    # Dictionary where the distances are the keys and the value is a list of couples where the keyed
    # distance is the distance between them
    max_distance = args.max_distance
    distances_couples = {idx: [] for idx in range(1, max_distance + 1)}

    start = time.time()
    for idx, (source, destination, distance) in enumerate(test_dataset, 1):
        distances_couples[distance].append((source, destination, distance))
        print_progress_bar(
            idx, dataset_len, time.time() - start, prefix=f'Collecting distances\' couples', length=50)

    rnd_generator = random.Random()
    randomed_couples_per_distance = []
    couples_amount_per_distance = args.amount_per_distance
    for idx in range(1, max_distance + 1):
        couples_amount_per_distance = args.amount_per_distance
        if len(distances_couples[idx]) < couples_amount_per_distance:
            couples_amount_per_distance = len(distances_couples[idx])
        randomed_couples_per_distance.extend(
            rnd_generator.sample(distances_couples[idx], couples_amount_per_distance))

    randomed_couples_per_distance_df = pd.DataFrame.from_records(randomed_couples_per_distance,
                                                                 columns=['source', 'destination', 'min_distance'])

    randomed_couples_per_distance_df.to_csv(
        path.join(path.dirname(args.test), path.splitext(path.basename(args.test))[0] + "_astar.csv"),
        header=True, index=False, sep='\t')
