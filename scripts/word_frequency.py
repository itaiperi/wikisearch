import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from scripts.utils import print_progress_bar, Cache
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_TEXT
from wikisearch.embeddings import FastText
from wikisearch.utils.mongo_handler import MongoHandler

import matplotlib.pyplot as plt


def choose_bin(word_occurrences, bin_edges):
    if word_occurrences == bin_edges[-1]:
        # Edge case, in which right edge of last bin equals to occurrences, because of different behavior for last bin
        return len(bin_edges) - 1
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= word_occurrences < bin_edges[i + 1]:
            return i


def draw_pie(ax, data, labels, title):
    wedges, _ = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
    plt.legend(labels, loc="upper right", bbox_to_anchor=(1.5, 1))
    ax.set_title(title)


def draw_bar(ax, indices, data, labels, title, xlabel, ylabel):
    ax.bar(indices, data)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=30)
    for index, value, s in zip(indices, np.array(data) + 10, [str(value) for value in hist]):
        ax.text(index, value, s)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default=sys.path[0], help="Path to save file to (default: script location)")

    pages_handler = MongoHandler(WIKI_LANG, PAGES)
    all_pages = pages_handler.get_all_documents()
    cache = Cache()
    pages_text = cache["word_frequency_pages_text"]
    if pages_text is None:
        pages_text = []
        all_pages_len = all_pages.count()
        start_time = time.time()

        for i, page in enumerate(all_pages):
            if ENTRY_TEXT in page:
                pages_text.append(FastText.tokenize_text(page[ENTRY_TEXT]))
            print_progress_bar(i, all_pages_len, time.time() - start_time, length=50)
        print()
        cache["word_frequency_pages_text"] = pages_text

    pages_text_flattened = [word for page in pages_text for word in page]
    pages_text_no_repeats = [set(page_text) for page_text in pages_text]
    pages_text_no_repeats_flattened = [word for page in pages_text_no_repeats for word in page]
    counter = Counter(pages_text_flattened)
    counter_no_repeats = Counter(pages_text_no_repeats_flattened)
    total_words = sum(counter.values())
    total_words_no_repeats = len(counter_no_repeats)

    bin_edges = [1, 10, 100, 500, 1000, 5000, 10000, max(counter_no_repeats.values())]
    bin_labels = [f"{left}-{right}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    bin_indices = list(range(len(bin_edges) - 1))

    # Set bin for each word, and words for each bin. word_to_bin is set based on with-repeats-in-document counts
    # bin_to_words - choose bin per counts with repeat, according to counts with repeat
    # bin_to_words_no_repeat - choose bin per counts without repeat, according to counts without repeat
    # bin_to_words_no_repeat_count_with_repeats - choose bin per counts without repeat, according to counts with repeat
    word_to_bin = {}
    bin_to_words = defaultdict(dict)
    bin_to_words_no_repeat = defaultdict(dict)
    bin_to_words_no_repeat_count_with_repeat = defaultdict(dict)
    for k, v in counter_no_repeats.items():
        bin = choose_bin(counter[k], bin_edges)
        bin_no_repeats = choose_bin(v, bin_edges)
        word_to_bin[k] = bin
        bin_to_words[bin][k] = counter[k]
        bin_to_words_no_repeat[bin_no_repeats][k] = v
        bin_to_words_no_repeat_count_with_repeat[bin_no_repeats][k] = counter[k]

    size_factor = 1.25
    fig = plt.figure(figsize=(16 * size_factor, 9 * size_factor))
    fig.suptitle("Binning according to # of documents a word appears in")
    plt.subplots_adjust(hspace=0.4, wspace=0.5, bottom=0, left=0.075)

    # Draw bar diagrams and pie charts
    ax = plt.subplot(2, 2, 1)
    hist, _ = np.histogram(list(counter_no_repeats.values()), bins=bin_edges)
    draw_bar(ax, bin_indices, hist, bin_labels, "Number of words appearing\nin # of documents",
             "Number of documents a word appears in", "Number of words in bin")

    ax = plt.subplot(2, 2, 2)
    hist = [sum(bin_to_words_no_repeat_count_with_repeat[index].values()) for index in bin_indices]
    draw_bar(ax, bin_indices, hist, bin_labels, "Volume (with repeats) of words\nappearing in # of documents",
             "Number of documents a word appears in", "Volume of appearances for words in bin")

    ax = plt.subplot(2, 2, 3)
    pie_data = [len(bin_to_words_no_repeat[index]) for index in bin_indices]
    pie_labels = [label + f" ({pie_data[index] / total_words_no_repeats * 100:.1f}%)"
                  for label, index in zip(bin_labels, bin_indices)]
    draw_pie(ax, pie_data, pie_labels, "Percentage of words\nappearing in # of documents")

    ax = plt.subplot(2, 2, 4)
    pie_data = [sum(bin_to_words_no_repeat_count_with_repeat[index].values()) for index in bin_indices]
    pie_labels = [label + f" ({pie_data[index] / total_words * 100:.1f}%)"
                  for label, index in zip(bin_labels, bin_indices)]
    draw_pie(ax, pie_data, pie_labels, "Percentage of volume of words\nappearing in # of document")
    plt.savefig(os.path.join(args.out, 'word_frequency_documents_bins.jpg'))

    fig = plt.figure(figsize=(16 * size_factor, 9 * size_factor))
    fig.suptitle("Binning according to # of times a word appears in all text")
    plt.subplots_adjust(hspace=0.4, wspace=0.5, bottom=0, left=0.075)

    ax = plt.subplot(2, 2, 1)
    hist, _ = np.histogram(list(counter.values()), bins=bin_edges)
    draw_bar(ax, bin_indices, hist, bin_labels, "Number of words appearing\ntotal of # times",
             "Number of times a word appears", "Number of words in bin")

    ax = plt.subplot(2, 2, 2)
    hist = [sum(bin_to_words[index].values()) for index in bin_indices]
    draw_bar(ax, bin_indices, hist, bin_labels,
             "Volume (with repeats) of words\nappearing total # of times in all text",
             "Number of times word appears in all text", "Volume of appearances for words in bin")

    ax = plt.subplot(2, 2, 3)
    pie_data = [len(bin_to_words[index]) for index in bin_indices]
    pie_labels = [label + f" ({pie_data[index] / total_words_no_repeats * 100:.1f}%)"
                  for label, index in zip(bin_labels, bin_indices)]
    draw_pie(ax, pie_data, pie_labels, "Percentage of words\nappearing in # of times")

    ax = plt.subplot(2, 2, 4)
    pie_data = [sum(bin_to_words[index].values()) for index in bin_indices]
    pie_labels = [label + f" ({pie_data[index] / total_words * 100:.1f}%)"
                  for label, index in zip(bin_labels, bin_indices)]
    draw_pie(ax, pie_data, pie_labels, "Percentage of volume of words\nappearing # of times in all text")
    plt.savefig(os.path.join(args.out, 'word_frequency_appearances_bins.jpg'))
