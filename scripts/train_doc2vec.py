import argparse
import time

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from scripts.utils import print_progress_bar, Cache
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TITLE, ENTRY_TEXT
from wikisearch.embeddings import Doc2Vec as Doc2VecEmbeddings
from wikisearch.utils.mongo_handler import MongoHandler


class Doc2VecProgressCallback(CallbackAny2Vec):
    """
    Class that is used as a callback for the training progress of Doc2Vec.
    """
    def __init__(self):
        self.epoch = 0
        self.start = time.time()

    def on_epoch_end(self, model):
        """
        Monitor training progress on each epoch's end.
        :param model: model that is trained
        :return: None
        """
        self.epoch += 1
        print_progress_bar(self.epoch, model.epochs, time.time() - self.start, prefix="Doc2Vec model training", length=50)


parser = argparse.ArgumentParser()
parser.add_argument('--vocab', '-v', required=True, help='Path to vocabulary file')
parser.add_argument('--out', '-o', required=True, help='Path for model output file')
parser.add_argument('--epochs', '-e', default=50, help='Number of training epochs')

args = parser.parse_args()

cache = Cache()

mongo = MongoHandler(WIKI_LANG, PAGES)
pages_cursor = mongo.get_all_documents()
start = time.time()

# Sample pages as a training dataset (or whole database, if isn't too big) for Doc2Vec and corpus.
# Sample random entries from database
# train_pages = mongo._collection.aggregate([{'$sample': {'size': 10000}}])
train_pages = cache['train_pages'] or []
if not train_pages:
    for i, page in enumerate(pages_cursor, 1):
        train_pages.append(TaggedDocument(Doc2VecEmbeddings.tokenize_text(page[ENTRY_TEXT]), page[ENTRY_TITLE]))
        print_progress_bar(i, pages_cursor.count(), time.time() - start, prefix="Building TaggedDocuments", length=50)
cache['train_pages'] = train_pages

start = time.time()
model = Doc2Vec(vector_size=300, min_count=2, epochs=args.epochs, callbacks=[Doc2VecProgressCallback()])
with open(args.vocab) as f:
    # Build vocabulary for model
    model.build_vocab(train_pages, progress_per=10)
print(f'-TIME- Defining model and building vocabulary took {time.time() - start:.1f}s')

model.train(train_pages, total_examples=len(train_pages), epochs=model.epochs)
model.save(args.out)
print('Finished Doc2Vec training')
