#!/bin/sh

TEST_CASES=`for case in \`cat ./test_nn_passed_lists.txt\`; do echo -n $case " "; done`; pytest -v $TEST_CASES