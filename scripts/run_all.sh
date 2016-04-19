#! /bin/sh
dropbox.py stop
./run_branch.py --skip-dropbox --branch-pred-type off
./run_branch.py --skip-dropbox --branch-pred-type always
./run_branch.py --skip-dropbox --branch-pred-type on
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipsec-encryption.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipsec-encryption.click --combine-cpu-gpu
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipv4-router-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 64 -l default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 72 -l default.py ipv6-router-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 72 -l default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 72 -l default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipsec-encryption-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipsec-encryption-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 64 -l default.py ipsec-encryption-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipsec-encryption.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipsec-encryption-gpuonly.click
dropbox.py start
