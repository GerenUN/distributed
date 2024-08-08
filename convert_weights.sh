git clone --depth 1 --branch v0.11.0 https://github.com/NVIDIA/TensorRT-LLM.git
hf_weights_dir="/opt/tritonserver/Mistral-7B-Instruct-v0.2"
hf_converted_weights="converted_weights/Mistral-7B-Instruct-v0.2"
trt_engine_1gpu="trt-engines/mistral/fp16/1-gpu"

python3 TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir $hf_weights_dir \
        --dtype float16 \
        --output_dir  $hf_converted_weights