import numpy as np
from typing import List

def memory_usage_bytes(file_name:str) -> None:
    nums:List[int] = []
    with open(file_name, "r") as f:
        for line in f:
            nums.append(int(line))
    
    np_arr = np.array(nums)
    np_arr = np_arr[np_arr!=0] 
    print(f"Mean: {np.mean(np_arr)/1024**3:.4f}Gb  Max: {np.max(np_arr)/1024**3:.4f}Gb")


def memory_usage_mib(file_name:str) -> None:
    nums:List[int] = []
    with open(file_name, "r") as f:
        for line in f:
            nums.append(int(line)*1.049e6)
    
    np_arr = np.array(nums)
    np_arr = np_arr[np_arr!=0] 
    print(f"Mean: {np.mean(np_arr)/1024**3:.4f}Gb  Max: {np.max(np_arr)/1024**3:.4f}Gb")


if __name__ == "__main__":

    memory_usage_bytes("test_system_memory.log")
    memory_usage_mib("test_gpu_memory.log")

