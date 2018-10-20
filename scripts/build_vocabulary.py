import argparse
import os
import time

from importlib import import_module
from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TEXT
from wikisearch.utils.mongo_handler import MongoHandler

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', required=True, help='Vocabulary output path')
parser.add_argument('--suffix', help='Suffix to output filename. Template: <db>_<suffix>.vocab')
parser.add_argument('--embedding', required=True, choices=['Doc2Vec'])

args = parser.parse_args()

# Dynamically load the relevant embedding class. This is used for tokenization later on!
embedding_module = import_module('.'.join(['wikisearch', 'embeddings', args.embedding.lower()]), package='wikisearch')
embedding_class = getattr(embedding_module, args.embedding)

mongo_handler = MongoHandler(WIKI_LANG, PAGES)

entire_start = time.time()
vocab = set()
all_pages_cursor = mongo_handler.get_all_pages()
num_of_pages = all_pages_cursor.count()
for i, page in enumerate(all_pages_cursor):
    page_text = embedding_class.tokenize_text(page[ENTRY_TEXT])
    vocab.update(page_text)
    print_progress_bar(i + 1, num_of_pages, time.time() - entire_start, prefix="Vocabulary build", length=50)

start = time.time()
vocab_size = len(vocab)
with open(os.path.join(args.out, WIKI_LANG + ('_' + args.suffix if args.suffix else '') + '.vocab'), 'w') as f:
    for i, word in enumerate(sorted(vocab)):
        f.write(word + '\n')
        print_progress_bar(i + 1, vocab_size, time.time() - start, prefix="Vocabulary write", length=50)

print(f"Building vocabulary took {time.time() - entire_start:.1f} seconds.")
print(f"Total vocabulary size is {len(vocab)}")
