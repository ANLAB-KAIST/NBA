#!/bin/bash
if [ $# -lt 1 ]; then
    echo You need to specify \"on\" or \"off\" as an argument.
    exit 1
fi
NUM_PROC=`cat /sys/devices/system/cpu/present | sed 's/-/ /g' | awk '{print $2}'`
if [ "$1" == "off" ]; then
    echo 'CPU 0 cannot be disabled. (always enabled)'
fi
for ((i=1;i<=$NUM_PROC;i++)); do
    ENABLED=`cat /sys/devices/system/cpu/cpu$i/online`
    if [ "$1" == "on" -a "$ENABLED" == "0" ]; then
	echo Enabling CPU $i ..
	echo 1 > /sys/devices/system/cpu/cpu$i/online
    fi
    if [ "$1" == "off" -a "$ENABLED" == "1" ]; then
	echo Disabling CPU $i ..
	echo 0 > /sys/devices/system/cpu/cpu$i/online
    fi
done
