# proxy set for downloading dataset error
# export http_proxy="http://child-jf.intel.com:912"
# export https_proxy="http://child-jf.intel.com:912"

# specify the model path
model='/media/newdrive2/huggingface/llama3-8b'

python llama3_ft.py -m ${model} 2>&1 | tee llama3_ft.log