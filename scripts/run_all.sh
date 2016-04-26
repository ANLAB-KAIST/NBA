#! /bin/sh
dropbox.py stop

# Figure 1 and 22
./run_branch.py --skip-dropbox --branch-pred-type off
./run_branch.py --skip-dropbox --branch-pred-type always
./run_branch.py --skip-dropbox --branch-pred-type on  # unused in the paper

# Figure 21
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipsec-encryption.click --combine-cpu-gpu

# Figure 23
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipsec-encryption.click --combine-cpu-gpu

# Figure 25
./run_app_perf.py --prefix latency -b bin-backup/main -p 64,256,1500 -l default.py l2fwd-echo.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipv4-router-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 64 -l default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 72 -l default.py ipv6-router-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 72 -l default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 72 -l default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipsec-encryption-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipsec-encryption-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 64 -l default.py ipsec-encryption-gpuonly.click

# Figure 24 and 26
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipsec-encryption.click --combine-cpu-gpu

# Figure 26
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipsec-encryption-gpuonly.click

# Figure 2 and 27
# NOTE: Finding max thruput should be done manually from lbratio results.
./run_loadbalancer_per_cpu_ratio.py -b bin-backup/main -p 64 default.py ipv4-router-lbratio.click 0 1 0.1
./run_loadbalancer_per_cpu_ratio.py -b bin-backup/main -p 64 default.py ipv6-router-lbratio.click 0 1 0.1
./run_loadbalancer_per_cpu_ratio.py -b bin-backup/main -p 64,256,1024 default.py ipsec-encryption-lbratio.click 0 1 0.1
./run_app_perf.py --prefix alb -b bin-backup/main --timeout 60 -p 64 default.py ipv4-router.click
./run_app_perf.py --prefix alb -b bin-backup/main --timeout 60 -p 64 default.py ipv6-router.click
./run_app_perf.py --prefix alb -b bin-backup/main --timeout 60 -p 64,256,1024 default.py ipsec-encryption.click
#./run_loadbalancer_per_cpu_ratio.py -b bin-backup/main --trace default.py ipsec-encryption-lbratio.click 0 1 0.1
#./run_app_perf.py --prefix alb -b bin-backup/main --trace default.py ipsec-encryption.click

dropbox.py start
