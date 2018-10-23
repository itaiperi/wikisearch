import argparse
import time

from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES
from wikisearch.embeddings import Word2VecAverage

parser = argparse.ArgumentParser()
parser.add_argument('--in', '-i', dest='inp', required=True)
parser.add_argument('--out', '-o', required=True)

args = parser.parse_args()

missing_counter = 0
dataset_size = 0
with open(args.inp, 'r') as in_file:
    dataset_size = len(list(in_file.readlines())) - 1

with open(args.inp, 'r') as in_file:
    with open(args.out, 'w') as out_file:
        embedder = Word2VecAverage(WIKI_LANG, PAGES)
        start = time.time()
        for i, line in enumerate(in_file.readlines()):
            if i == 0:
                out_file.write(line)
                continue
            source_title, dest_title, min_distance = line.split('\t')
            source_tensor = embedder.embed(source_title)
            dest_tensor = embedder.embed(dest_title)

            if source_tensor.size() and dest_tensor.size():
                out_file.write(f'{source_title}\t{dest_title}\t{min_distance}')
            else:
                missing_counter += 1
            print_progress_bar(i, dataset_size, time.time() - start, length=50)
print(f'Missing counter: {missing_counter}')

