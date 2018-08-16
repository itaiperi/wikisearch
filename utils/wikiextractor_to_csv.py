import argparse
import json
import os
from urllib.parse import unquote
import re
import time

ENTRY_TITLE = "title"
ENTRY_URL = "url"
ENTRY_ID = "id"
ENTRY_TEXT = "text"
ENTRY_LINKS = "links"
CSV_SEPARATOR = "\t"
LINKS_SEPARATOR = ";"


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", required=True, help="Wikiextractor results directory")
parser.add_argument("-o", "--out", required=True, help="Output dir")

args = parser.parse_args()
# Find all links which are internal (skip http and https links)
internal_link_regex = re.compile("<a href=\"(?!(?:https?://))(.*?)\">.*?</a>")
start = time.time()

with open(os.path.join(args.out, "wikisearch.csv"), "w") as f_entries:
    f_entries.write(CSV_SEPARATOR.join([ENTRY_TITLE, ENTRY_ID, ENTRY_URL, ENTRY_TEXT, ENTRY_LINKS]) + "\n")

    for d in sorted(os.listdir(args.dir)):
        d_path = os.path.join(args.dir, d)
        for filename in sorted(os.listdir(d_path)):
            filepath = os.path.join(d_path, filename)
            with open(filepath) as f:
                for line in f.readlines():
                    entry = json.loads(line)
                    # Turn all percentage symbols (e.g %20) into regular ascii (e.g space)
                    entry[ENTRY_TEXT] = unquote(entry[ENTRY_TEXT])
                    # TODO Need to take care ALSO of http and https links, so they will be substituted by only the description
                    # TODO Need to take care of escaped characters such as \" etc. (need to see which exist)
                    # TODO Need to take care of parentheses, other symbols.
                    # TODO Need to check if we need to remove stuff like paris trips etc.
                    links = set(internal_link_regex.findall(entry[ENTRY_TEXT]))
                    entry[ENTRY_TEXT] = internal_link_regex.sub("\g<1>", entry[ENTRY_TEXT]).replace("\n", " ")
                    f_entries.write(CSV_SEPARATOR.join([entry[ENTRY_TITLE], entry[ENTRY_ID], entry[ENTRY_URL], entry[ENTRY_TEXT],
                                                        LINKS_SEPARATOR.join(links)]) + "\n")

print("Processing took {} seconds".format(int(time.time() - start)))