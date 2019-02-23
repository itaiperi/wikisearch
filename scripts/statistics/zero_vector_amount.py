import argparse
import os

import torch

from scripts.loaders import load_embedder_by_name
from wikisearch.consts.mongo import ENTRY_TITLE
from wikisearch.consts.nn import EMBEDDING_VECTOR_SIZE
from wikisearch.embeddings import AVAILABLE_EMBEDDINGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, choices=AVAILABLE_EMBEDDINGS)
    parser.add_argument("-o", "--output", help="Path to the zero vectors pages file")
    args = parser.parse_args()

    embedder = load_embedder_by_name(args.embedding)

    zeros_vectors = [page[ENTRY_TITLE]
                     for page in embedder._mongo_handler_embeddings.get_all_documents()
                     if torch.all(embedder._decode_vector(page[embedder.__class__.__name__.lower()]) == torch.zeros(EMBEDDING_VECTOR_SIZE[embedder.type]))]

    zeros_vectors = "\n".join(zeros_vectors)
    with open(os.path.join(args.output, "zero_vectors.txt"), "w", encoding="utf8") as f:
        f.write(zeros_vectors)

    print(f"Found {len(zeros_vectors)}")
