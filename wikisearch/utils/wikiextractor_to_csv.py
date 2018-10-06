import argparse
import json
from urllib.parse import unquote
import re
import time
from wikisearch.consts.mongo import *


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", required=True, help="Wikiextractor results directory")
parser.add_argument("-o", "--out", required=True, help="Output dir")

args = parser.parse_args()
# Find all links which are internal (skip http and https links)
internal_link_regex = re.compile("<a href=\"(?!(?:https?://))(?P<title>.*?)\">(?P<text>.*?)</a>")
start = time.time()

with open(os.path.join(args.out, "wikisearch.csv"), "w") as f_entries:
    f_entries.write(CSV_SEPARATOR.join([ENTRY_TITLE, ENTRY_PID, ENTRY_TEXT, ENTRY_LINKS]) + "\n")

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
                    # Extract internal links from text
                    links = set([m.group('title') for m in internal_link_regex.finditer(entry[ENTRY_TEXT])])
                    # Substitute line breaks with spaces
                    entry[ENTRY_TEXT] = entry[ENTRY_TEXT].replace("\n", " ")
                    # Substitute internal links with their representative text
                    entry[ENTRY_TEXT] = internal_link_regex.sub("\g<text>", entry[ENTRY_TEXT])

                    # Write entry to CSV
                    f_entries.write(CSV_SEPARATOR.join([entry[ENTRY_TITLE], entry[ENTRY_PID], entry[ENTRY_TEXT],
                                                        LINKS_SEPARATOR.join(links)]) + "\n")

print("Processing took {} seconds".format(int(time.time() - start)))
