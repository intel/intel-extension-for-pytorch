#!/bin/bash

function ver_compare() {
    VER_MAJOR_CUR=$(echo $1 | cut -d "." -f 1)
    VER_MINOR_CUR=$(echo $1 | cut -d "." -f 2)
    VER_PATCH_CUR=$(echo $1 | cut -d "." -f 3)
    VER_MAJOR_REQ=$(echo $2 | cut -d "." -f 1)
    VER_MINOR_REQ=$(echo $2 | cut -d "." -f 2)
    VER_PATCH_REQ=$(echo $2 | cut -d "." -f 3)
    RET=0
    if [[ ${VER_MAJOR_CUR} -lt ${VER_MAJOR_REQ} ]]; then
        RET=1
    else
        if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
           [[ ${VER_MINOR_CUR} -lt ${VER_MINOR_REQ} ]]; then
            RET=2
        else
            if [[ ${VER_MAJOR_CUR} -eq ${VER_MAJOR_REQ} ]] &&
               [[ ${VER_MINOR_CUR} -eq ${VER_MINOR_REQ} ]] &&
               [[ ${VER_PATCH_CUR} -lt ${VER_PATCH_REQ} ]]; then
                RET=3
            fi
        fi
    fi
    echo ${RET}
}

LIBSTDCPP_SYS=""
LIBSTDCPP_ARR=()
while read -r line; do
    LIBSTDCPP_ARR+=($line)
done < <(find /usr -regextype sed -regex ".*libstdc++\.so\.[[:digit:]]*\.[[:digit:]]*\.[[:digit:]]*" 2>/dev/null)
if [ ${#LIBSTDCPP_ARR[@]} -gt 1 ]; then
    for value in "${LIBSTDCPP_ARR[@]}"; do
        if [[ "$value" =~ ^/usr/lib(64)*/ ]]; then
            LIBSTDCPP_SYS=$value
        fi
    done
else
    LIBSTDCPP_SYS=${LIBSTDCPP_ARR[0]}
fi
LIBSTDCPP_ACTIVE=${LIBSTDCPP_SYS}
if [ ! -z ${CONDA_PREFIX} ]; then
    LIBSTDCPP_CONDA=$(find ${CONDA_PREFIX}/lib -regextype sed -regex ".*libstdc++\.so\.[[:digit:]]*\.[[:digit:]]*\.[[:digit:]]*" 2>/dev/null)
    LIBSTDCPP_VER_SYS=$(echo ${LIBSTDCPP_SYS} | sed "s/.*libstdc++.so.//")
    LIBSTDCPP_VER_CONDA=$(echo ${LIBSTDCPP_CONDA} | sed "s/.*libstdc++.so.//")
    VER_COMP=$(ver_compare ${LIBSTDCPP_VER_CONDA} ${LIBSTDCPP_VER_SYS})
    if [[ ${VER_COMP} -gt 0 ]]; then
        LIBSTDCPP_ACTIVE=${LIBSTDCPP_SYS}
    else
        LIBSTDCPP_ACTIVE=${LIBSTDCPP_CONDA}
    fi
fi
echo ${LIBSTDCPP_ACTIVE}
