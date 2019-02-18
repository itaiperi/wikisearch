import argparse
import itertools
import os
import subprocess
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from wikisearch.heuristics.nn_archs import NN_ARCHS


def product_dict(d):
    return [dict(zip(d, x)) for x in itertools.product(*d.values())]


# NN archs comparison
nn_archs = product_dict({"--arch": NN_ARCHS, "-b": [256], "-e": [200], "--crit": ["MSELoss"], "--opt": ["SGD"],
                         "--lr": [1e-3], "--embedding": ["FastTextTitle"]})

# Embeddings comparison
embeddings = product_dict({"--arch": ["EmbeddingsDistance2"], "-b": [256], "-e": [200], "--crit": ["MSELoss"],
                           "--opt": ["SGD"], "--lr": [1e-3], "--embedding": ["FastTextTitle", "Word2VecTitle"]})

# Criterions comparison
criterions = product_dict({"--arch": ["EmbeddingsDistance2"], "-b": [256], "-e": [200], "--crit": ["MSELoss"],
                           "--opt": ["SGD"], "--lr": [1e-3], "--embedding": ["FastTextTitle"]}) + \
             product_dict({"--arch": ["EmbeddingsDistance2"], "-b": [256], "-e": [200], "--crit": ["AsymmetricMSELoss"],
                           "--alphas": ["1 3", "1 5", "1 10"], "--opt": ["SGD"], "--lr": [1e-3],
                           "--embedding": ["FastTextTitle"]})

# SGD with and without momentum
sgd = product_dict({"--arch": ["EmbeddingsDistance2"], "-b": [256], "-e": [200], "--crit": ["MSELoss"],
                    "--opt": ["SGD"], "--lr": [1e-3], "--sgd-momentum": [0, 0.9], "--embedding": ["FastTextTitle"]})

# Adam, for comparison with SGD
adam = product_dict({"--arch": ["EmbeddingsDistance2"], "-b": [256], "-e": [200], "--crit": ["MSELoss"],
                     "--opt": ["Adam"], "--lr": [1e-3], "--embedding": ["FastTextTitle"]})

all_models_params = nn_archs + embeddings + criterions + sgd + adam


def train_and_test_model(model_params):
    params_str = " ".join([" ".join([str(k), str(v)]) for k, v in model_params.items()])
    train_command = f"python /media/Data/Projects/wikisearch/scripts/embeddings_nn.py {params_str}"
    test_command = f"python /media/Data/Projects/wikisearch/scripts/statistics/calculate_distances_statistics.py -m {params['-o']} -df {params['-te']}"
    print(train_command)
    subprocess.call(train_command, shell=True, stdout=open(os.path.join(os.path.dirname(model_params['-o']), 'train.log'), 'w'))
    print(test_command)
    subprocess.call(test_command, shell=True, stdout=open(os.path.join(os.path.dirname(model_params['-o']), 'test.log'), 'w+'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="dataset_dir", help="Directory where dataset train, val, test files are")
    parser.add_argument('-o', '--out', required=True, help="Directory to which models will be written")
    parser.add_argument("-w", "--num-workers", default=1, type=int)
    args = parser.parse_args()

    train_file = os.path.join(args.dataset_dir, "train.csv")
    validation_file = os.path.join(args.dataset_dir, "validation.csv")

    for params in all_models_params:
        model_dir = '_'.join([params['--embedding'], params['--arch'], params['--crit'], params['--opt']] +
                             ([params['--alphas']] if '--alphas' in params else []))
        model_dir = model_dir.lower().replace(' ', '_')
        params['-tr'] = train_file
        params['-te'] = validation_file
        params['-o'] = os.path.join(args.out, model_dir, 'model.pth')

    pool = Pool(min(args.num_workers, cpu_count() - 1))
    pool.map(train_and_test_model, all_models_params)
