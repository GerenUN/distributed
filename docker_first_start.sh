cd code
git clone --depth 1 --branch v0.11.0 https://github.com/NVIDIA/TensorRT-LLM.git

hf_weights_dir="Mistral-7B-Instruct-v0.2"
hf_converted_weights="converted_weights/Mistral-7B-Instruct-v0.2"
trt_engine_1gpu="trt-engines/mistral/fp16/1-gpu"

python3 TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir $hf_weights_dir \
        --dtype float16 \
        --output_dir  $hf_converted_weights


trtllm-build --checkpoint_dir $hf_converted_weights \
        --workers 8 \
        --gpt_attention_plugin float16 \
        --max_input_len 2048 \
        --max_output_len 2048 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --output_dir $trt_engine_1gpu \
        # --use_fused_mlp \
        # --reduce_fusion disable \
        # --use_paged_context_fmha enable \
        # --max_batch_size $max_batch_size \
        # --max_num_tokens $max_num_tokens \
        # --multiple_profiles enable


printf  "\n\n    *********  Testing Engine...   *********\n\n"

python3 TensorRT-LLM/examples/run.py \
    --engine_dir=$trt_engine_1gpu \
    --max_output_len 128 \
    --tokenizer_dir $hf_weights_dir \
    --input_text "How do I count in French ? 1 un"