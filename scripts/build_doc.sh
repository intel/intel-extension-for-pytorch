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
if [ ${DEVICE} == "gpu" ]; then
    MDCONF="tutorials/features/advanced_configuration.md"
    if [ ! -f ${MDCONF} ]; then
        echo "${MDCONF} not found. Quit."
        exit 1
    fi
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
    parse_example "../examples/cpu/training/single_instance_training_fp32.py" ${MDEXAMPLE} "(marker_train_single_fp32_complete)" "python"
    parse_example "../examples/cpu/training/single_instance_training_bf16.py" ${MDEXAMPLE} "(marker_train_single_bf16_complete)" "python"
    parse_example "../examples/cpu/training/distributed_data_parallel_training.py" ${MDEXAMPLE} "(marker_train_ddp_complete)" "python"
    parse_example "../examples/cpu/inference/python/resnet50_eager_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp32)" "python"
    parse_example "../examples/cpu/inference/python/bert_eager_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_imp_fp32)" "python"
    parse_example "../examples/cpu/inference/python/resnet50_torchscript_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_ts_fp32)" "python"
    parse_example "../examples/cpu/inference/python/bert_torchscript_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_ts_fp32)" "python"
    parse_example "../examples/cpu/inference/python/resnet50_torchdynamo_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_dynamo_fp32)" "python"
    parse_example "../examples/cpu/inference/python/bert_torchdynamo_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_dynamo_fp32)" "python"
    parse_example "../examples/cpu/inference/python/resnet50_eager_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_bf16)" "python"
    parse_example "../examples/cpu/inference/python/bert_eager_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_imp_bf16)" "python"
    parse_example "../examples/cpu/inference/python/resnet50_torchscript_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_ts_bf16)" "python"
    parse_example "../examples/cpu/inference/python/bert_torchscript_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_ts_bf16)" "python"
    parse_example "../examples/cpu/inference/python/resnet50_torchdynamo_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_dynamo_bf16)" "python"
    parse_example "../examples/cpu/inference/python/bert_torchdynamo_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_dynamo_bf16)" "python"
    parse_example "../examples/cpu/features/fast_bert/fast_bert_inference_bf16.py" ${MDEXAMPLE} "(marker_feature_fastbert_bf16)" "python"
    parse_example "../examples/cpu/inference/python/int8_quantization_static.py" ${MDEXAMPLE} "(marker_int8_static)" "python"
    parse_example "../examples/cpu/inference/python/int8_quantization_dynamic.py" ${MDEXAMPLE} "(marker_int8_dynamic)" "python"
    parse_example "../examples/cpu/inference/python/int8_deployment.py" ${MDEXAMPLE} "(marker_int8_deploy)" "python"
    parse_example "../examples/cpu/features/llm/llm_optimize.py" ${MDEXAMPLE} "(marker_llm_optimize)" "python"
    parse_example "../examples/cpu/features/llm/llm_optimize_smoothquant.py" ${MDEXAMPLE} "(marker_llm_optimize_sq)" "python"
    parse_example "../examples/cpu/features/llm/llm_optimize_woq.py" ${MDEXAMPLE} "(marker_llm_optimize_woq)" "python"
    parse_example "../examples/cpu/inference/cpp/example-app.cpp" ${MDEXAMPLE} "(marker_cppsdk_sample)" "cpp"
    parse_example "../examples/cpu/inference/cpp/CMakeLists.txt" ${MDEXAMPLE} "(marker_cppsdk_cmake)" "cmake"
	VER_TRANS=$(python ../tools/yaml_utils.py -f ../dependency_version.yml -d transformers -k version)
	sed -i "s/<VER_TRANSFORMERS>/${VER_TRANS}/g" ${MDEXAMPLE}

    cp tutorials/features/fast_bert.md tutorials/features/fast_bert.md.bk
    parse_example "../examples/cpu/features/fast_bert/fast_bert_inference_bf16.py" tutorials/features/fast_bert.md "(marker_feature_fastbert_bf16)" "python"
    cp tutorials/features/graph_optimization.md tutorials/features/graph_optimization.md.bk
    parse_example "../examples/cpu/features/graph_optimization/fp32_bf16.py" tutorials/features/graph_optimization.md "(marker_feature_graph_optimization_fp32_bf16)" "python"
    parse_example "../examples/cpu/features/graph_optimization/int8.py" tutorials/features/graph_optimization.md "(marker_feature_graph_optimization_int8)" "python"
    parse_example "../examples/cpu/features/graph_optimization/folding.py" tutorials/features/graph_optimization.md "(marker_feature_graph_optimization_folding)" "python"
elif [[ ${DEVICE} == "gpu" ]]; then
    parse_example "../examples/gpu/training/single_instance_training_fp32.py" ${MDEXAMPLE} "(marker_train_single_fp32_complete)" "python"
    parse_example "../examples/gpu/training/single_instance_training_bf16.py" ${MDEXAMPLE} "(marker_train_single_bf16_complete)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_imperative_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp32)" "python"
    parse_example "../examples/gpu/inference/python/bert_imperative_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_imp_fp32)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_torchscript_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_rn50_ts_fp32)" "python"
    parse_example "../examples/gpu/inference/python/bert_torchscript_mode_inference_fp32.py" ${MDEXAMPLE} "(marker_inf_bert_ts_fp32)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_imperative_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_bf16)" "python"
    parse_example "../examples/gpu/inference/python/bert_imperative_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_imp_bf16)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_torchscript_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_rn50_ts_bf16)" "python"
    parse_example "../examples/gpu/inference/python/bert_torchscript_mode_inference_bf16.py" ${MDEXAMPLE} "(marker_inf_bert_ts_bf16)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_imperative_mode_inference_fp16.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp16)" "python"
    parse_example "../examples/gpu/inference/python/bert_imperative_mode_inference_fp16.py" ${MDEXAMPLE} "(marker_inf_bert_imp_fp16)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_torchscript_mode_inference_fp16.py" ${MDEXAMPLE} "(marker_inf_rn50_ts_fp16)" "python"
    parse_example "../examples/gpu/inference/python/bert_torchscript_mode_inference_fp16.py" ${MDEXAMPLE} "(marker_inf_bert_ts_fp16)" "python"
    parse_example "../examples/gpu/inference/python/resnet50_imperative_mode_inference_fp32_alt.py" ${MDEXAMPLE} "(marker_inf_rn50_imp_fp32_alt)" "python"
    # parse_example "../examples/gpu/inference/python/int8_calibration_static_imperative.py" ${MDEXAMPLE} "(marker_int8_static_imperative)" "python"
    parse_example "../examples/gpu/inference/python/int8_quantization_static.py" ${MDEXAMPLE} "(marker_int8_static)" "python"
    # parse_example "../examples/gpu/inference/python/int8_deployment.py" ${MDEXAMPLE} "(marker_int8_deploy)" "python"
    parse_example "../examples/gpu/inference/cpp/example-app/example-app.cpp" ${MDEXAMPLE} "(marker_cppsdk_sample_app)" "cpp"
    parse_example "../examples/gpu/inference/cpp/example-app/CMakeLists.txt" ${MDEXAMPLE} "(marker_cppsdk_cmake_app)" "cmake"
    parse_example "../examples/gpu/inference/cpp/example-usm/example-usm.cpp" ${MDEXAMPLE} "(marker_cppsdk_sample_usm)" "cpp"
    parse_example "../examples/gpu/inference/cpp/example-usm/CMakeLists.txt" ${MDEXAMPLE} "(marker_cppsdk_cmake_usm)" "cmake"

    cp ${MDCONF} tutorials/features/advanced_configuration.md.bk
    sed -i "/^| [[:alnum:]_-]/d" ${MDCONF}
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
cp tutorials/features/graph_capture.md tutorials/features/graph_capture.md.bk
parse_example "../examples/cpu/features/graph_capture.py" tutorials/features/graph_capture.md "(marker_feature_graph_capture)" "python"
cp tutorials/features/int8_recipe_tuning_api.md tutorials/features/int8_recipe_tuning_api.md.bk
parse_example "../examples/cpu/features/int8_recipe_tuning/int8_autotune.py" tutorials/features/int8_recipe_tuning_api.md "(marker_feature_int8_autotune)" "python"

make clean
make html

mv tutorials/features/graph_capture.md.bk tutorials/features/graph_capture.md
mv tutorials/features/int8_recipe_tuning_api.md.bk tutorials/features/int8_recipe_tuning_api.md
mv tutorials/examples.md.bk tutorials/examples.md
if [[ ${DEVICE} == "cpu" ]]; then
    mv tutorials/features/fast_bert.md.bk tutorials/features/fast_bert.md
    mv tutorials/features/graph_optimization.md.bk tutorials/features/graph_optimization.md
elif [[ ${DEVICE} == "gpu" ]]; then
    rm -rf xml
    mv tutorials/features/advanced_configuration.md.bk tutorials/features/advanced_configuration.md
fi
