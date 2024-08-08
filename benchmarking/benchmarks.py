import argparse
import json
import random
import time
from typing import List, Optional, Tuple
import resource

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser

from benchmarking.dataset import sample_requests
# TODO: warm up iters
# TODO: peak memory and avg memory usage tests
# TODO: latency

print(f"memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3:.4f} Gb")


# sample different lengths
requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                            args.output_len)

if args.backend == "vllm":
    elapsed_time = run_vllm(
        requests, args.model, args.tokenizer, args.quantization,
        args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
        args.trust_remote_code, args.dtype, args.max_model_len,
        args.enforce_eager, args.kv_cache_dtype,
        args.quantization_param_path, args.device,
        args.enable_prefix_caching, args.enable_chunked_prefill,
        args.max_num_batched_tokens, args.distributed_executor_backend,
        args.gpu_memory_utilization, args.download_dir, args.load_format)
    
def main():
    pass



def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
    )

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start



if __name__ == "__main__":
    main()