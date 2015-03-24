#!/bin/bash
NUM_PROC=`cat /sys/devices/system/cpu/present | sed 's/-/ /g' | awk '{print $2}'`
for ((i=1;i<=$NUM_PROC;i++)); do
    ENABLED=`cat /sys/devices/system/cpu/cpu$i/online`
    if [ "$ENABLED" == "0" ]; then
	echo Enabling CPU $i ..
	echo 1 > /sys/devices/system/cpu/cpu$i/online
    fi
done
