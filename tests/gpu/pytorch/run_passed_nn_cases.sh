#!/bin/sh

cat test_nn_passed_lists.txt | grep -v '#' | tr '\n' ' ' | xargs pytest -v
