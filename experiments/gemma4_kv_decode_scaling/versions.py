"""Print the vLLM, torch and Triton versions of the Pod image."""
import torch
import triton
import vllm

print("vllm", vllm.__version__)
print("torch", torch.__version__)
print("triton", triton.__version__)
