#!/bin/bash

# This script has to be run in intel-extension-for-pytorch/docs directory.
# When compiling doc for intel-extension-for-pytorch GPU, activating DPCPP compiler env and oneMKL env is required.

if [ $# -ne 1 ]; then
    echo "Usage: bash ../scripts/$0 <DEVICE>"
    echo "DEVICE: cpu | gpu"
    exit 1
fi
DEVICE=$1

MDEXAMPLE="tutorials/examples.md"
if [ ! -f ${MDEXAMPLE} ]; then
    echo "${MDEXAMPLE} not found. Quit."
    echo "Please run this script in docs directory."
    exit 1
fi
MDCONF="tutorials/features/advanced_configuration.md"
if [ ! -f ${MDCONF} ]; then
    echo "${MDCONF} not found. Quit."
    exit 1
fi

set -e
set -x

# for gpu
parse_build_options() {
    SETTINGSCPP=$1
    DOCFILE=$2
    txt="| ------ | ------ | ------ |"
    while read -r line; do
        option=${line}
        option=${option#option(}
        option=${option%)}
        option=${option// \"/\"}
        option=${option//\" /\"}
        key=$(echo ${option} | cut -d "\"" -f 1)
        description=$(echo ${option} | cut -d "\"" -f 2)
        description=${description/\*/\\\\*}
        value=$(echo ${option} | cut -d "\"" -f 3)
        txt="${txt}\n| ${key} | ${value} | ${description} |"
    done < <(grep "^option(" ${SETTINGSCPP})
    while read -r line; do
        option=${line}
        option=${option#set(}
        option=${option%)}
        option=${option# CACHE STRING }
        option=${option// \"/\"}
        option=${option//\" /\"}
        num_fields=$(echo ${option} | grep -o "\"" | wc -l)
        if [[ ${num_fields} -eq 4 ]]; then
            key=$(echo ${option} | cut -d "\"" -f 1)
            value=$(echo ${option} | cut -d "\"" -f 2)
            description=$(echo ${option} | cut -d "\"" -f 4)
            description=${description/\*/\\\\*}
            txt="${txt}\n| ${key} | \"${value}\" | ${description} |"
        fi
    done < <(grep "^set(" ${SETTINGSCPP})
    while read -r line; do
        option=${line}
        option=${option#cmake_dependent_option(}
        option=${option%)}
        option=${option// \"/\"}
        option=${option//\" /\"}
        num_fields=$(echo ${option} | grep -o "\"" | wc -l)
        if [[ ${num_fields} -eq 4 ]]; then
            key=$(echo ${option} | cut -d "\"" -f 1)
            description=$(echo ${option} | cut -d "\"" -f 2)
            description=${description/\*/\\\\*}
            value1=$(echo ${option} | cut -d "\"" -f 3)
            value2=$(echo ${option} | cut -d "\"" -f 5)
            condition=$(echo ${option} | cut -d "\"" -f 4)
            txt="${txt}\n| ${key} | ${value1} if \"${condition}\" is ON, otherwise ${value2} | ${description} |"
        fi
    done < <(grep "^cmake_dependent_option(" ${SETTINGSCPP})
    ln=$(grep "| \*\*Build Option\*\* | \*\*Default<br>Value\*\* | \*\*Description\*\* |" -n ${DOCFILE} | cut -d ":" -f 1)
    ln=$((ln+1))
    sed -i "${ln} i ${txt}" ${DOCFILE}
}

# for gpu
parse_launch_options() {
    SETTINGSCPP=$1
    DOCFILE=$2
    MARKER=$3
    ln_start=-1
    ln_end=-1
    while read -r line; do
        if [[ ${ln_start} -eq -1 ]]; then
            ln_start=$(echo ${line} | cut -d ":" -f 1)
        else
            ln_end=$(echo ${line} | cut -d ":" -f 1)
        fi
    done < <(grep -n ${MARKER} ${SETTINGSCPP})
    ln_start=$((ln_start+1))
    ln_end=$((ln_end-1))
    key=""
    content=""
    txt="| ------ | ------ | ------ |"
    for i in $(seq ${ln_start} ${ln_end})
    do
        line=$(head -n ${i} ${SETTINGSCPP} | tail -n 1)
        line=${line/ \*   /}
        if [[ ${line} =~ ^[a-zA-Z0-9] ]]; then
            if [[ ${key} != "" ]]; then
                content=${content// | /|}
                value=$(echo ${content} | cut -d "|" -f 1)
                value=${value//Default = /}
                description=$(echo ${content} | cut -d "|" -f 2)
                description=${description//\*/\\\\*}
                txt="${txt}\n| ${key} | ${value} | ${description} |"
                key=""
                content=""
            fi
            key=${line}
            key=${key//:/}
        else
            content="${content} ${line}"
        fi
        if [[ ${i} -eq ${ln_end} ]]; then
            content=${content// | /|}
            value=$(echo ${content} | cut -d "|" -f 1)
            value=${value//Default = /}
            description=$(echo ${content} | cut -d "|" -f 2)
            description=${description//\*/\\\\*}
            txt="${txt}\n| ${key} | ${value} | ${description} |"
        fi
    done
    IDEN=""
    if [[ ${MARKER} == "==========ALL==========" ]]; then
        IDEN="CPU, GPU"
    fi
    if [[ ${MARKER} == "==========GPU==========" ]]; then
        IDEN="GPU ONLY"
    fi
    if [[ ${MARKER} == "==========EXP==========" ]]; then
        IDEN="Experimental"
    fi
    ln=$(grep "| \*\*Launch Option<br>${IDEN}\*\* | \*\*Default<br>Value\*\* | \*\*Description\*\* |" -n ${DOCFILE} | cut -d ":" -f 1)
    ln=$((ln+1))
    sed -i "${ln} i ${txt}" ${DOCFILE}
}

# for cpu and gpu
parse_example() {
    EXAMPLE=$1
    DOCFILE=$2
    MARKER=$3
    SYNTAX=$4
    ln_start=-1
    ln_end=-1
    while read -r line; do
        if [[ ${ln_start} -eq -1 ]]; then
            ln_start=$(echo ${line} | cut -d ":" -f 1)
        else
            ln_end=$(echo ${line} | cut -d ":" -f 1)
        fi
    done < <(grep -n ${MARKER} ${DOCFILE})
    ln_start=$((ln_start+1))
    ln_end=$((ln_end-1))
    if [ ${ln_end} -gt ${ln_start} ]; then
        sed -i "${ln_start},${ln_end}d" ${DOCFILE}
    fi
    sed -i "${ln_start}i \`\`\`${SYNTAX}\\n\`\`\`" ${DOCFILE}
    ln_start=$((ln_start+1))
    code=$(cat ${EXAMPLE} | sed 's/\\n/\\\\n/g' | sed -E ':a;N;$!ba;s/\r{0,1}\n/\\n/g')
    sed -i "${ln_start}i ${code}" ${DOCFILE}
}

cp ${MDEXAMPLE} tutorials/examples.md.bk
if [[ ${DEVICE} == "cpu" ]]; then
    parse_example "../examples/cpu/training/python-scripts/single_instance_training_fp32.py" ${MDEXAMPLE} "(marker_train_single_fp32_complete)" "python"
    parse_example "../examples/cpu/training/python-scripts/single_instance_training_bf16.py" ${MDEXAMPLE} "(marker_train_single_bf16_complete)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/resnet50_imperative_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp32)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/bert_imperative_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_imp_fp32)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/resnet50_torchdynamo_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_dynamo_fp32)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/bert_torchdynamo_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_dynamo_fp32)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/resnet50_imperative_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_bf16)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/bert_imperative_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_imp_bf16)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/bert_fast_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_fast_bf16)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/int8_calibration_static.py" ${MDEXAMPLE} "(marker_int8_static)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/int8_calibration_dynamic.py" ${MDEXAMPLE} "(marker_int8_dynamic)" "python"
    parse_example "../examples/cpu/inference/python/python-scripts/int8_deployment.py" ${MDEXAMPLE} "(marker_int8_deploy)" "python"
    parse_example "../examples/cpu/inference/cpp/example-app.cpp" ${MDEXAMPLE} "(marker_cppsdk_sample)" "cpp"
    parse_example "../examples/cpu/inference/cpp/CMakeLists.txt" ${MDEXAMPLE} "(marker_cppsdk_cmake)" "cmake"

    cp tutorials/features/fast_bert.md tutorials/features/fast_bert.md.bk
    parse_example "../examples/cpu/inference/python/python-scripts/bert_fast_inference_bf16.py" tutorials/features/fast_bert.md "(marker_inf_bert_fast_bf16)" "python"
    cp tutorials/features/graph_optimization.md tutorials/features/graph_optimization.md.bk
    parse_example "../examples/cpu/features/graph_optimization/fp32_bf16.py" tutorials/features/graph_optimization.md "(marker_feature_graph_optimization_fp32_bf16)" "python"
    parse_example "../examples/cpu/features/graph_optimization/int8.py" tutorials/features/graph_optimization.md "(marker_feature_graph_optimization_int8)" "python"
    parse_example "../examples/cpu/features/graph_optimization/folding.py" tutorials/features/graph_optimization.md "(marker_feature_graph_optimization_folding)" "python"
elif [[ ${DEVICE} == "gpu" ]]; then
    parse_example "../examples/gpu/training/python-scripts/single_instance_training_fp32.py" ${MDEXAMPLE} "(marker_train_single_fp32_complete)" "python"
    parse_example "../examples/gpu/training/python-scripts/single_instance_training_bf16.py" ${MDEXAMPLE} "(marker_train_single_bf16_complete)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/resnet50_imperative_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp32)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/bert_imperative_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_imp_fp32)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/resnet50_imperative_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_bf16)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/bert_imperative_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_imp_bf16)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/resnet50_imperative_mode_inference_fp16.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp16)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/bert_imperative_mode_inference_fp16.py" ${MDEXAMPLE} "(marker_inf_bert_imp_fp16)" "python"
    parse_example "../examples/gpu/inference/python/python-scripts/resnet50_imperative_mode_inference_fp32_alt.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp32_alt)" "python"

    cp ${MDCONF} tutorials/features/advanced_configuration.md.bk
    #sed -i "/^| [[:alnum:]_-]/d" ${MDCONF}
    parse_build_options "../cmake/gpu/Options.cmake" ${MDCONF}
    parse_launch_options "../csrc/gpu/utils/Settings.cpp" ${MDCONF} "==========ALL=========="
    parse_launch_options "../csrc/gpu/utils/Settings.cpp" ${MDCONF} "==========GPU=========="
    parse_launch_options "../csrc/gpu/utils/Settings.cpp" ${MDCONF} "==========EXP=========="

    if [ -d ../csrc/include/xpu_bk ]; then
        rm -rf ../csrc/include/xpu
        mv ../csrc/include/xpu_bk ../csrc/include/xpu
    fi
    cp -r ../csrc/include/xpu ../csrc/include/xpu_bk
    find ../csrc/include/xpu -name "*.h" -exec sed -i "s/IPEX_API //g" {} \;
    if [ -d xml ]; then
        rm -rf xml
    fi
    doxygen
    rm -rf ../csrc/include/xpu
    mv ../csrc/include/xpu_bk ../csrc/include/xpu
fi

make clean
make html

mv tutorials/examples.md.bk tutorials/examples.md
if [[ ${DEVICE} == "cpu" ]]; then
    mv tutorials/features/fast_bert.md.bk tutorials/features/fast_bert.md
    mv tutorials/features/graph_optimization.md.bk tutorials/features/graph_optimization.md
elif [[ ${DEVICE} == "gpu" ]]; then
    rm -rf xml
    mv tutorials/features/advanced_configuration.md.bk tutorials/features/advanced_configuration.md
fi

LN=$(grep "searchtools.js" -n _build/html/search.html | cut -d ":" -f 1)
sed -i "${LN}i \ \ \ \ <script src=\"_static/js/theme.js\"></script>" _build/html/search.html
sed -i "${LN}i \ \ \ \ <script src=\"_static/sphinx_highlight.js?v=dc90522c\"></script>" _build/html/search.html
sed -i "${LN}i \ \ \ \ <script src=\"_static/doctools.js?v=9a2dae69\"></script>" _build/html/search.html
sed -i "${LN}i \ \ \ \ <script src=\"_static/documentation_options.js?v=fc837d61\"></script>" _build/html/search.html
sed -i "${LN}i \ \ \ \ <script src=\"_static/_sphinx_javascript_frameworks_compat.js?v=2cd50e6c\"></script>" _build/html/search.html
sed -i "${LN}i \ \ \ \ <script src=\"_static/jquery.js?v=5d32c60e\"></script>" _build/html/search.html
sed -i "${LN}i \ \ \ \ <\!\-\-[if lt IE 9]><script src=\"_static/js/html5shiv.min.js\"></script><\![endif]\-\->" _build/html/search.html
