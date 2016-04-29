#! /bin/sh
dropbox.py stop
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipv4-router.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipv6-router.click --combine-cpu-gpu
./run_app_perf.py --prefix scalability -b bin-backup/main --num-cores 1,2,4,7 -p 64 default.py ipsec-encryption.click --combine-cpu-gpu
dropbox.py start
