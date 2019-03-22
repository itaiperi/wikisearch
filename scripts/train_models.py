import argparse
import itertools
import json
import os
import subprocess
import sys
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import pandas as pd
import tabulate


def product_dict(d):
    d_listed_values = {k: [v] if type(v) != list else v for k, v in d.items()}
    return [dict(zip(d_listed_values, x)) for x in itertools.product(*d_listed_values.values())]


def remove_duplicates(l):
    return [dict(t) for t in {tuple(d.items()) for d in l}]


def train_and_test_model(model_params):
    params_str = " ".join([" ".join([str(k), str(v)]) for k, v in model_params.items()])
    train_command = f"python {os.path.join(sys.path[0], 'embeddings_nn.py')} {params_str}"
    test_command = f"python {os.path.join(sys.path[0], 'statistics/calculate_distances_statistics.py')} -m" \
        f"{os.path.join(model_params['-o'], 'model.pth')} -df {model_params['-te']}"
    subprocess.call(train_command, shell=True, stdout=open(os.path.join(model_params['-o'], 'train.log'), 'w'))
    subprocess.call(test_command, shell=True, stdout=open(os.path.join(model_params['-o'], 'test.log'), 'w+'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="dataset_dir", help="Directory where dataset train, val, test files are")
    parser.add_argument('-i', '--inp', help="Json file which includes all experiments parameters")
    parser.add_argument('-o', '--out', help="Directory to which models will be written (Default: dataset directory)")
    parser.add_argument("-w", "--num-workers", default=1, type=int)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.dataset_dir

    # Import parameters from json file, and expand with product_dict
    with open(args.inp) as params_f:
        all_models_params = remove_duplicates([params for multi_params in json.load(params_f)
                                               for params in product_dict(multi_params)])

    params_names = set([key for params in all_models_params for key in params.keys()])
    columns_order = ['--embedding', '--arch', '--crit', '--alphas', '--opt', '--sgd-momentum', '--lr', '-b', '-e']
    models_df = pd.DataFrame(all_models_params)
    models_df = models_df[[column for column in columns_order if column in params_names]]
    print(tabulate.tabulate(models_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    print(f"Total of {len(models_df)} experiments")

    # Add needed parameters, which are not included in the json file
    train_file = os.path.join(args.dataset_dir, "train.csv")
    validation_file = os.path.join(args.dataset_dir, "validation.csv")

    for params in all_models_params:
        model_dir = '_'.join([params['--embedding'], params['--arch'], params['--crit'], params['--opt'],
                              str(params['--lr']), str(params['-b'])] +
                             ([params['--alphas']] if '--alphas' in params else []) +
                             ([str(params['--sgd-momentum'])] if '--sgd-momentum' in params else []))
        model_dir = model_dir.lower().replace(' ', '_')
        model_dir = os.path.join(args.out, model_dir)

        if os.path.exists(model_dir) and os.listdir(model_dir):
            # Filter models that were already run in the past (directory exists and is not empty!
            continue
        os.makedirs(model_dir, exist_ok=True)
        params['-tr'] = train_file
        params['-te'] = validation_file
        params['-o'] = model_dir

    # Only models that haven't been run before will have -o parameter, and we run only them!
    all_models_params = [params for params in all_models_params if '-o' in params]

    # Train and test!
    if args.num_workers == 1:
        for params in all_models_params:
            train_and_test_model(params)
    else:
        pool = Pool(min(args.num_workers, cpu_count() - 1))
        pool.map(train_and_test_model, all_models_params)
