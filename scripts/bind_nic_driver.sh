#!/bin/bash

# Declaring subroutine
function print_usage()
{
    # -e enables tab & newline in echo command.
    echo -e "Usage: [-h] [-b driver_to_bind] [-p dpdk_path]"
    echo -e "Options:\n \t-h\t: print usage\n \t-b\t: specify driver module to use (must be loaded ahead)\n \t-p\t: path of dpdk bind script"
    echo -e "Example:\n \tsudo ./bind_nic_driver.sh -b igb_uio -p ~/dpdk-1.6.0.r2/tools/igb_uio_bind.py\n"
}

# main start
echo -e "\nNOTICE:\n 1 Run this script using sudo.\n 2. Kernel modules to be used must be loaded before executing this script.\n"

# Checking the # of argument 
if [ $# -eq 0 ]; then
    print_usage
    exit
fi

# Parsing argument; bash getopts do not support long option names.. 
while getopts "hb:p:" arg; do
    case $arg in
        h)
            print_usage
            exit
        ;;
        b)
            driver_to_bind=$OPTARG
        ;;
        p)
            dpdk_tool_path=$OPTARG
        ;;
        \?)
            print_usage
            exit
        ;;
    esac 
done

# To check if variable is set. ref: http://stackoverflow.com/a/13864829/2013586
# ${parameter:+[word]} : If parameter is unset or null, null shall be substituted
if [ -z ${dpdk_tool_path+x} ] || [ -z ${driver_to_bind+x} ]; then
    echo "Required variables are not set."
    print_usage
    exit
fi

# set driver module to bind
python $dpdk_tool_path --status | while read -r line 
do
    echo "$line" | grep -q "82599ES"
    if [ $? -eq 0 ]; then
        # Using "sed" first to remove substring in front of port name, and then using cut to remove substring behind of.
        #port_name=$(echo "$line" | sed -n -e 's/^.*\(if=\)//p' | cut -f1 -d" ")
        #echo $port_name
        #python $dpdk_tool_path --bind=igb_uio $port_name

        # Using "cut" to get mac address
        mac_addr=$(echo "$line" | cut -f1 -d" ")
        echo $mac_addr
        python $dpdk_tool_path -u $mac_addr
        python $dpdk_tool_path -b $driver_to_bind $mac_addr
    fi
done

