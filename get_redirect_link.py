import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xml', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--title')
    group.add_argument('-f', '--file')

    args = parser.parse_args()

    redirect_re = re.compile("#REDIRECT \[\[(.*)]]")
    title_re = re.compile("<title>(.*)</title>")

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            redirect_entries = [line.replace('\n', '') for line in f.readlines()]
    else:
        redirect_entries = [args.title]

    with open(args.xml, 'r', encoding='utf-8') as f:
        found = False
        title = None
        for line in f.readlines():
            title_matches = title_re.findall(line)
            if title_matches:
                title = title_matches[0]
                if title in redirect_entries:
                    found = True

            elif found:
                redirect_matches = redirect_re.findall(line)
                if redirect_matches:
                    redirect = redirect_matches[0]
                    found = False
                    print(title, redirect)
