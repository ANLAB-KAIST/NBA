#! /bin/sh
./run_throughput.py -p 64,1500 --comp-batch-sizes 1 rss.py ipv4-router-cpuonly.click
./run_throughput.py -p 64,1500 --comp-batch-sizes 1 rss.py ipv6-router-cpuonly.click
./run_throughput.py -p 64,1500 --comp-batch-sizes 1 rss.py ipsec-encryption-cpuonly.click
./run_throughput.py -p 64,1500 --comp-batch-sizes 1 rss.py kargus_ids-cpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py ipv4-router-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py ipv6-router-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py ipsec-encryption-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py kargus_ids-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py ipv4-router-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py ipv6-router-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py ipsec-encryption-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py kargus_ids-gpuonly.click
