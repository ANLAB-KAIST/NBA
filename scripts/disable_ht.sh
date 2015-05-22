#!/bin/sh
cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list |
sort -u |
while read sibs
do
    case "$sibs" in
            *,*)
                    oldIFS="$IFS"
                    IFS=",$IFS"
                    set $sibs
                    IFS="$oldIFS"

                    shift
                    while [ "$1" ]
                    do
                            echo Disabling CPU $1 ..
                            echo 0 > /sys/devices/system/cpu/cpu$1/online
                            shift
                    done
                    ;;
            *)
                    ;;
    esac
done
