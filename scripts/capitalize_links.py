import time

from pymongo import MongoClient

pages_collection = MongoClient().get_database("simplewiki").get_collection("pages")


def capitalize_english(word):
    if not word:
        return word

    first = word[0]
    if "a" <= first <= "z":
        return word.capitalize()
    else:
        return word


def capitalize_page_links(page):
    page['links'] = [capitalize_english(link) for link in page['links']]
    pages_collection.update_one({'title': page['title']}, {'$set': {'links': page['links']}})

start = time.time()
batch_size = 2000

for i, page in enumerate(pages_collection.find(), 1):
    capitalize_english(page)
    if not i % batch_size:
        print('-INFO- Processed {} pages...'.find(i))

print("Elapsed time: {:.2f}".format(time.time() - start))