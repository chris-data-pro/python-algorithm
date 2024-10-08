import os
from pathlib import Path

"""

for k in d# Use Python/Go for writing "grind", a small application with features of
# grep and find.
#
# This is the specification. More features can be added later.
#
# Usage:
# grind [options]
#
# Options:
# -d, --dir DIRECTORY   Directory to start searching from.
# -m, --match MATCH     Only show the files that have MATCH in the filename.
# -r, --recursive       Search recursively. Don't follow symlinks.
#
# Example usage:
#
# $ grind -d /tmp/example/
# a/
# b/
# c/
# ipsum
# foobar
#
# $ grind -d /tmp/example -m foo
# foobar
#
# $ grind -d /tmp/example/ -m foo -r
# b/foo
# foobar
 
# The following directory structure is used in the examples above:
#
# for dir in a b c; do mkdir -p /tmp/example/$dir; done
# for file in a/bar b/foo c/zar foobar ipsum; do touch /tmp/example/$file; done
# tree /tmp/example/
#
# /tmp/example/
# |-- a
# |   `-- bar
# |-- b
# |   `-- foo
# |-- c
# |   `-- zar
# |-- foobar
# `-- ipsum

"""

if __name__ == '__main__':
    # create data path directory if not exists
    Path("/tmp/example/c/foobar").mkdir(parents=True, exist_ok=True)
    # print(os.listdir("/tmp/example/"))

    def grind(inputs):
        res = []
        inputs = inputs.split("-")
        for input in inputs:
            if not input:
                continue
            option = input.split(" ")[0]
            para = input.split(" ")[1]
            if option == 'd':
                res += os.listdir(para)
            if option == 'm':
                res = [r for r in res if para in r]
            if option == 'r':
                for r in res:
                    grind(input + r)
        return res

    print(grind("-d /tmp/example -m foo"))


