import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('title')
    args = parser.parse_args()
    q = []

    found = False
    with open('/home/itai/Downloads/simplewiki-20180901-pages-articles.xml', 'r') as f:
        for line in f.readlines():
            if line.find("<page>") > -1:
                q = []
            q.append(line)
            if line.find("<title>{}<".format(args.title)) > -1:
                found = True
            if line.find("</page>") > -1:
                if found:
                    for line in q:
                        print(line, end='')
                found = False
