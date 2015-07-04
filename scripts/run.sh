#! /bin/sh
./scripts/run_throughput.py -p 1024,1500 default.py ipsec-encryption-gpuonly.click
#./scripts/run_throughput.py -p 64,128,256,512,1024,1500 default.py ipv4-router-gpuonly.click
#./scripts/run_throughput.py -p 64,128,256,512,1024,1500 default.py ipv6-router-gpuonly.click
#./scripts/run_throughput.py -p 64,128,256,512,1024,1500 default.py ipsec-encryption-gpuonly.click
#./scripts/run_throughput.py -p 64,128,256,512,1024,1500 default.py kargus_ids-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py ipv4-router-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py ipv6-router-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py ipsec-encryption-gpuonly.click
#./run_throughput.py -p 64,1500 --comp-batch-sizes 16,32,64,128,256 rss.py kargus_ids-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py ipv4-router-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py ipv6-router-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py ipsec-encryption-gpuonly.click
#./run_throughput.py -p 64,1500 --coproc-ppdepths 8,16,32,64,128 rss.py kargus_ids-gpuonly.click
