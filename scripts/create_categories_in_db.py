import argparse
import os
import sys
import time
from collections import Counter

import matplotlib.pyplot as plt

from scripts.utils import print_progress_bar
from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_CATEGORIES, ENTRY_REDIRECT_TO, CATEGORIES, ENTRY_TITLE
from wikisearch.utils.mongo_handler import MongoHandler

REPORT_RESOLUTION = 100


def categories_pages_coverage(categories, pages_categories_list):
    count = 0
    uncovered_pages = []
    for page_categories in pages_categories_list:
        covered = len(categories & page_categories) > 0
        count += covered
        if not covered:
            uncovered_pages.append(page_categories)

    return count, uncovered_pages


def choose_categories(pages, percentage):
    # Take only pages which are not redirects, and that actually have categories
    pages_with_categories = [set(page[ENTRY_CATEGORIES]) for page in pages
                             if (ENTRY_REDIRECT_TO not in page and len(page[ENTRY_CATEGORIES]))]
    num_pages = len(pages_with_categories)
    categories_counter = Counter([category for categories in pages_with_categories for category in categories])
    total_categories = sum(categories_counter.values())

    covered_pages_counter, i = 0, 0
    # taken_categories is used to see how many categories are required for full coverage, not only required percentage
    taken_categories = set()
    # chosen_categories is the set only of the categories required to cover percentage of the pages
    chosen_categories = None
    exact_coverage = None
    coverage_percentages = []
    # uncovered_pages is used to track pages which haven't been covered by a category yet. Initially, that's all pages
    uncovered_pages = pages_with_categories
    start_time = time.time()
    while covered_pages_counter < num_pages and len(categories_counter):
        max_category = categories_counter.most_common(1)[0][0]
        del categories_counter[max_category]
        taken_categories.add(max_category)
        added_covered_pages, uncovered_pages = categories_pages_coverage(taken_categories, uncovered_pages)
        covered_pages_counter += added_covered_pages
        if covered_pages_counter / num_pages >= percentage and chosen_categories is None:
            chosen_categories = taken_categories.copy()
            exact_coverage = covered_pages_counter / num_pages
        i += 1
        if i % REPORT_RESOLUTION == 0:
            coverage_percentages.append(covered_pages_counter / num_pages)
        print_progress_bar(covered_pages_counter, num_pages, time.time() - start_time,
                           length=50, prefix="Categories Choosing")
    if i % REPORT_RESOLUTION != 0:
        coverage_percentages.append(covered_pages_counter / num_pages)

    print(
        f"Chose {len(chosen_categories)} categories covering {exact_coverage * 100:.1f}% of the DB")
    return chosen_categories, coverage_percentages, len(taken_categories)


def main(percentage):
    pages_mongo = MongoHandler(WIKI_LANG, PAGES)
    categories_mongo = MongoHandler(WIKI_LANG, CATEGORIES)

    all_pages = list(pages_mongo.get_all_documents())
    chosen_categories, coverage_percentages_history, total_categories = choose_categories(all_pages, percentage)

    plt.figure()
    indices = list(range(0, len(coverage_percentages_history) * REPORT_RESOLUTION, REPORT_RESOLUTION)) + [total_categories]
    plt.plot(indices, [0] + coverage_percentages_history)
    plt.grid(True)
    plt.axvline(len(chosen_categories), linestyle="--")
    plt.title("Number of categories chosen to cover percentage of pages")
    plt.xlabel("# Categories")
    plt.ylabel("% of Pages Covered")
    plt.savefig(os.path.join(sys.path[0], 'categories_coverage.jpg'))
    categories_mongo.update_page({ENTRY_TITLE: CATEGORIES, CATEGORIES: sorted(chosen_categories)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--percentage', required=True, type=float, help="Percentage of pages to cover with categories")
    args = parser.parse_args()

    main(args.percentage)
