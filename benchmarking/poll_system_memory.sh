while true; do 
free -b | grep  -v total | grep -v Swap | awk '{print $3}' >> $1_system_memory.log
sleep $2
done