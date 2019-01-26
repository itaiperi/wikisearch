import argparse
import json
import time
from importlib import import_module
from os import path

import pandas as pd
import tabulate
import torch.utils.data

from scripts.utils import print_progress_bar
from wikisearch.astar import Astar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, CSV_SEPARATOR
from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.consts.statistics_column_names import *
from wikisearch.costs.uniform_cost import UniformCost
from wikisearch.embeddings import EMBEDDINGS_MODULES
from wikisearch.graph import WikiGraph
from wikisearch.heuristics.nn_archs import EmbeddingsDistance
from wikisearch.heuristics.nn_heuristic import NNHeuristic
from wikisearch.strategies import DefaultAstarStrategy
from wikisearch.utils.clean_data import tokenize_title

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to the model file. When running from linux - '
                                              'notice to not put a \'/\' after the file name')
    parser.add_argument('-df', '--dataset_file', help='Path to a dataset file')
    args = parser.parse_args()

    model_dir_path = path.dirname(args.model)
    model_file_name = path.splitext(path.basename(args.model))[0]

    # Loads dynamically the relevant embedding class
    with open(path.join(model_dir_path, f"{model_file_name}.meta")) as f:
        model_metadata = json.load(f)
    embedding = model_metadata['embedder']['type']
    embedding_module = import_module(
        '.'.join(['wikisearch', 'embeddings', EMBEDDINGS_MODULES[embedding]]),
        package='wikisearch')
    embedding_class = getattr(embedding_module, embedding)
    embedder = embedding_class(WIKI_LANG, PAGES)

    # Loads the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingsDistance(EMBEDDING_VECTOR_SIZE).to(device)
    model.load_state_dict(torch.load(args.model, map_location=None if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # Loads the dataset file
    dataset = pd.read_csv(args.dataset_file, sep=CSV_SEPARATOR).values

    # Prepare the statistics table
    pd.set_option('display.max_columns', 10)
    pd.set_option('precision', 2)
    statistics_df = pd.DataFrame(columns=[SRC_NODE, DST_NODE, BFS_DIST, NN_DIST])
    cost = UniformCost()
    heuristic = NNHeuristic(model, embedder)
    strategy = DefaultAstarStrategy()
    graph = WikiGraph(WIKI_LANG)
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
                    NN_DIST: model(embedder.embed(source).unsqueeze(0), embedder.embed(destination).unsqueeze(0)).item()
                    # ASTAR_DIST: astar_distance
                }, ignore_index=True)
            print_progress_bar(idx, len(dataset), time.time() - start, prefix=f'Progress: ', length=50)

    # Print out the statistics to csv file
    statistics_file_path = path.join(model_dir_path, f"{model_file_name}.stats")
    statistics_df = statistics_df.rename(lambda col: col.replace(' ', '\n'), axis='columns')
    statistics_df.to_csv(statistics_file_path, sep=CSV_SEPARATOR, header=True, index=False)
