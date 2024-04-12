# proxy set for downloading dataset error
# export http_proxy="http://child-jf.intel.com:912"
# export https_proxy="http://child-jf.intel.com:912"

# specify the model path
model='path_to_llama3/llama3'

python llama3_ft.py -m ${model} 2>&1 | tee llama3_ft.log