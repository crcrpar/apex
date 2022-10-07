#!/bin/bash

set -euxo

# Ideally I want to pin clang-format version to 13 which I've used.
# https://pypi.org/project/lintrunner/ might do that but it looked too much to me.
find . -iregex '.*\.\(cpp\|h\|cu\|cuh\|cc\)$' ! -path "**/cudnn-frontend/**" ! -path "**/cutlass/**" -exec clang-format --style=Google -i {} +
