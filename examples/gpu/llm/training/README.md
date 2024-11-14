## LLM training

Here we provide the llm training examples from scratch.

Our examples integrate with the popular tools and libraries from the ecosystem:
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) for distributed training
- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) for [accessing model weights](https://huggingface.co/models)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) for training and evaluation datasets
- [Transformers](https://github.com/huggingface/transformers) for training (Trainer) and modeling script
- [Accelerate](https://github.com/huggingface/accelerate) for Multi-GPUs launch

### LLM training validated with Intel® Data Center Max 1550 GPU on Linux

**Note**:
Here we mainly focus on LLM training on multi-GPU, and provide examples for Mixtral 7B training from scratch.

| MODEL FAMILY | Verified < MODEL ID > (Hugging Face hub)| 
|---|:---:|
|[Mixtral 7B](./Mixtral/)| "mistralai/Mistral-7B-v0.1" | 
