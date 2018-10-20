import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('xml', help="Path to Wikidump XML file")
    parser.add_argument('title', help="Title of page to get XML of")
    args = parser.parse_args()
    q = []

    found = False
    with open(args.xml, 'r') as f:
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
