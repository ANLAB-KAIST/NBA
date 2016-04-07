#! /bin/sh
dropbox.py stop
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipv4-router-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 64 -l default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 64 -l default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 72 -l default.py ipv6-router-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main -p 72 -l default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/main.noreuse -p 72 -l default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/lmain -p 64 -l default.py ipsec-encryption-cpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/lmain -p 64 -l default.py ipsec-encryption-gpuonly.click
./run_app_perf.py --prefix latency -b bin-backup/lmain.noreuse -p 64 -l default.py ipsec-encryption-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main -p 64,128,256,512,1024,1500 default.py ipsec-encryption.click --combine-cpu-gpu
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipv4-router-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipv6-router-gpuonly.click
./run_app_perf.py --prefix thruput -b bin-backup/main.noreuse -p 64,128,256,512,1024,1500 default.py ipsec-encryption-gpuonly.click
dropbox.py start
