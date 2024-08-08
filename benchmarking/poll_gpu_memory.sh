while true; do 
nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}' >> $1_gpu_memory.log
sleep $2
done