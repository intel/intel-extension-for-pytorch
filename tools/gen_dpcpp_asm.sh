#!/bin/bash
if [ -z $1 ]
then
   echo "No source file provided "
   exit
else
      echo "Generating GPU assembly for $1"
fi

#Compile to generate fat assembly file
dpcpp -S $1 -o tmp.s

#Extract gpu bc file
clang-offload-bundler -type=s  -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown-sycldevice -inputs=tmp.s -outputs=cpu.s,gpu.s -unbundle
rm -rf cpu.s  tmp.s

#Generate gpu spirv file
sycl-post-link  --ompoffload-link-entries --ompoffload-sort-entries --ompoffload-make-globals-static --ir-output-only gpu.s -o gpu.bc
llvm-spirv -spirv-ext=+all gpu.bc -o gpu.spv
rm -rf gpu.s gpu.bc

#Set env variables
export IGC_EnableVISAOutput=1
export IGC_DumpToCurrentDir=1

#Invoke aot compiler
/rdrive/ref/opencl/runtime/linux/oclgpu/20.22.16952/ocloc -file gpu.spv -spirv_input -device cfl
rm PFWG_Kernel_after_0.dot PFWG_Kernel_after_0.ll PFWG_Kernel_before_0.dot PFWG_Kernel_before_0.ll gpu.spv gpu_Gen9core.bin gpu_Gen9core.gen
