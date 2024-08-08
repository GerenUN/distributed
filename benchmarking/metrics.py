import numpy as np
from typing import List
import logging

def memory_usage(file_name:str) -> None:
    nums:List[int] = []
    mode = "GPU" if "gpu" in file_name else "System"
    with open(file_name, "r") as f:
        for line in f:
            nums.append(int(line)*1.049e6 if mode == "GPU" else int(line))
    
    np_arr = np.array(nums)
    np_arr = np_arr[np_arr!=0] 
    assert np.min(np_arr) != 0, "there were 0's in the log after filtering"
    logging.info(f"{mode} memory usage Mean: {np.mean(np_arr)/1024**3:.4f}Gb  Max: {np.max(np_arr)/1024**3:.4f}Gb")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    memory_usage("test_system_memory.log")
    memory_usage("test_gpu_memory.log")

