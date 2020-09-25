#!/bin/bash

# test all other UT
# readlink -f $0

pytest -v example

# test all nn UT
cat test_nn_passed_lists.txt | while read line
do 
    # python -m pytest -v $line
    pytest -v $line
    echo ${line}
done
echo "Finish nn UT test in IPEX"
