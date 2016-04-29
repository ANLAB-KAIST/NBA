#! /bin/sh
dropbox.py stop
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix compbatching -b bin-backup/main --comp-batch-sizes 1,4,8,16,32,64 -p 64,256,1500 default.py ipsec-encryption.click --combine-cpu-gpu
dropbox.py start
