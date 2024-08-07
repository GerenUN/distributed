docker run --rm --ipc=host --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.4.1-devel-ubuntu22.04
docker run --rm --ipc=host --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

# ref
# https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file

docker cp ~/code/Mistral-7B-Instruct-v0.2 cbf3491ef577:/opt/tritonserver/code/

#  check installation 
python3 -c "import tensorrt_llm"

cd code

git clone https://github.com/NVIDIA/TensorRT-LLM.git

git clone --depth 1 --branch v0.11.0 https://github.com/NVIDIA/TensorRT-LLM.git
# pip install -r TensorRT-LLM/examples/llama/requirements.txt 
# pip install --upgrade protobuf

hf_weights_dir="Mistral-7B-Instruct-v0.2"
hf_converted_weights="converted_weights/Mistral-7B-Instruct-v0.2"
trt_engine_1gpu="trt-engines/mistral/fp16/1-gpu"

python3 TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir $hf_weights_dir \
        --dtype float16 \
        --output_dir  $hf_converted_weights

# trtllm-build --checkpoint_dir $hf_converted_weights \
#                 --use_gpt_attention_plugin float16  \
#                 --use_inflight_batching \
#                 --paged_kv_cache enable \
#                 --remove_input_padding enable\
#                 --use_gemm_plugin float16  \
#                 --output_dir $trt_engine_1gpu  \
#                 --max_input_len 2048 --max_output_len 512 \
#                 --use_rmsnorm_plugin float16  \
#                 --enable_context_fmha

# ref https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html#engine-building
# https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html#../advanced/gpt-attention.html#in-flight-batchingT
trtllm-build --model_config $model_cfg --use_fused_mlp --gpt_attention_plugin float16 --output_dir $engine_dir --max_batch_size $max_batch_size --max_input_len 2048 --max_output_len 2048 --reduce_fusion disable --workers $tp_size --max_num_tokens $max_num_tokens --use_paged_context_fmha enable --multiple_profiles enable

trtllm-build --checkpoint_dir $hf_converted_weights \
        --gpt_attention_plugin float16 \
        --max_input_len 2048 \
        --max_output_len 2048 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --output_dir $trt_engine_1gpu \
        --use_fused_mlp \
        --reduce_fusion disable \
        --use_paged_context_fmha enable \
        --max_batch_size $max_batch_size \
        --max_num_tokens $max_num_tokens \
        --multiple_profiles enable


python3 TensorRT-LLM/examples/run.py \
    --engine_dir=$trt_engine_1gpu \
    --max_output_len 128 \
    --tokenizer_dir $hf_weights_dir \
    --input_text "How do I count in French ? 1 un"