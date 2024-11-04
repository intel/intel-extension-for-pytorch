set -x
filenames=$(ls | grep -E test_.*\\.py)
#for filename in ${filenames}
#   do
#     echo -e "execute python ${filename}"
#     python ${filename} 2>&1 | tee "${filename}.log"
#   done

fail=$(grep -R "Testing model fail:" *.py.log | wc -l)
pass=$(grep -R "Testing model success:" *.py.log | wc -l)
echo ${pass}
echo ${fail}
total=$((${pass}+${fail}))
echo ${total}

passrate=$(echo "${pass} ${total}"| awk '{print $1/$2}')
echo $passrate
