import argparse
import itertools
import os
import subprocess
import sys
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import pandas as pd
import tabulate

# from scripts.utils import print_progress_bar
from wikisearch.heuristics.nn_archs import NN_ARCHS


def product_dict(d):
    return [dict(zip(d, x)) for x in itertools.product(*d.values())]


train_epochs = [120]

# Batch size comparison
batch_sizes = product_dict({"--arch": ["EmbeddingsDistance"], "-b": [64, 256, 1024], "-e": train_epochs,
                            "--crit": ["MSELoss"], "--opt": ["SGD"], "--lr": [1e-3], "--embedding": ["FastTextTitle"]})

# NN archs comparison
nn_archs = product_dict({"--arch": NN_ARCHS[:2], "-b": [256], "-e": train_epochs, "--crit": ["MSELoss"],
                         "--opt": ["SGD"], "--lr": [1e-3], "--embedding": ["FastTextTitle"]})

# Embeddings comparison
embeddings = product_dict({"--arch": ["EmbeddingsDistance"], "-b": [256], "-e": train_epochs, "--crit": ["MSELoss"],
                           "--opt": ["SGD"], "--lr": [1e-3], "--embedding": ["FastTextTitle", "Word2VecTitle"]})

# Categories, for comparison with "without categories"
categories = product_dict({"--arch": ["EmbeddingsDistanceCategoriesMultiHot"], "-b": [256], "-e": train_epochs,
                           "--crit": ["MSELoss"], "--opt": ["SGD"], "--lr": [1e-3],
                           "--embedding": ["FastTextTitleCategoriesMultiHot", "Word2VecTitleCategoriesMultiHot"]})

# Criterions comparison
criterions = product_dict({"--arch": ["EmbeddingsDistance"], "-b": [256], "-e": train_epochs, "--crit": ["MSELoss"],
                           "--opt": ["SGD"], "--lr": [1e-3], "--embedding": ["FastTextTitle"]}) + \
             product_dict({"--arch": ["EmbeddingsDistance"], "-b": [256], "-e": train_epochs,
                           "--crit": ["AsymmetricMSELoss"], "--alphas": ["1 3", "1 5", "1 10"], "--opt": ["SGD"],
                           "--lr": [1e-3], "--embedding": ["FastTextTitle"]})

# SGD with and without momentum
sgd = product_dict({"--arch": ["EmbeddingsDistance"], "-b": [256], "-e": train_epochs, "--crit": ["MSELoss"],
                    "--opt": ["SGD"], "--lr": [1e-3], "--sgd-momentum": [0, 0.9], "--embedding": ["FastTextTitle"]})

# Adam, for comparison with SGD
adam = product_dict({"--arch": ["EmbeddingsDistance"], "-b": [256], "-e": train_epochs, "--crit": ["MSELoss"],
                     "--opt": ["Adam"], "--lr": [1e-3], "--embedding": ["FastTextTitle"]})

all_models_params = batch_sizes + nn_archs + embeddings + criterions + sgd + adam + categories
# Drop duplicates
all_models_params = [dict(t) for t in {tuple(d.items()) for d in all_models_params}]


def train_and_test_model(model_params):
    params_str = " ".join([" ".join([str(k), str(v)]) for k, v in model_params.items()])
    train_command = f"python {os.path.join(sys.path[0], 'embeddings_nn.py')} {params_str}"
    test_command = f"python {os.path.join(sys.path[0], 'statistics/calculate_distances_statistics.py')} -m {model_params['-o']} -df {model_params['-te']}"
    subprocess.call(train_command, shell=True, stdout=open(os.path.join(os.path.dirname(model_params['-o']), 'train.log'), 'w'))
    subprocess.call(test_command, shell=True, stdout=open(os.path.join(os.path.dirname(model_params['-o']), 'test.log'), 'w+'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="dataset_dir", help="Directory where dataset train, val, test files are")
    parser.add_argument('-o', '--out', help="Directory to which models will be written (Default: dataset directory)")
    parser.add_argument("-w", "--num-workers", default=1, type=int)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.dataset_dir

    models_df = pd.DataFrame(all_models_params)
    models_df = models_df[['--embedding', '--arch', '--crit', '--alphas', '--opt', '--sgd-momentum', '--lr', '-b', '-e']]
    print(tabulate.tabulate(models_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print(f"Total of {len(models_df)} experiments")

    train_file = os.path.join(args.dataset_dir, "train.csv")
    validation_file = os.path.join(args.dataset_dir, "validation.csv")

    for params in all_models_params:
        model_dir = '_'.join([params['--embedding'], params['--arch'], params['--crit'], params['--opt'],
                              str(params['--lr']), str(params['-b'])] +
                             ([params['--alphas']] if '--alphas' in params else []) +
                             ([str(params['--sgd-momentum'])] if '--sgd-momentum' in params else []))
        model_dir = model_dir.lower().replace(' ', '_')
        model_dir = os.path.join(args.out, model_dir)
        model_path = os.path.join(model_dir, 'model.pth')

        if os.path.exists(model_dir):
            # Filter models that were already run in the past
            continue
        os.makedirs(model_dir)
        params['-tr'] = train_file
        params['-te'] = validation_file
        params['-o'] = model_path

    # Only models that haven't been run before will have -o parameter
    all_models_params = [params for params in all_models_params if '-o' in params]

    if args.num_workers == 1:
        for params in all_models_params:
            train_and_test_model(params)
    else:
        pool = Pool(min(args.num_workers, cpu_count() - 1))
        pool.map(train_and_test_model, all_models_params)
