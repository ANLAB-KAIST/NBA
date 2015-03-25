# NBA (Network Balancing Act)
"A High-performance packet processing framework for heterogeneous processors," [EuroSys 2015 paper](http://an.kaist.ac.kr/~sbmoon/paper/intl-conf/2015-eurosys-nba.pdf)

## Disclaimer

* The IDS source code is not available to the public, as it contains a derivation from industry-tranferred code from [Kargus](http://shader.kaist.edu/kargus/).
  - You could refer to other open-source code, such as [a GPU implementation of the Aho-Corasick algorithm in Snap](https://github.com/wbsun/g4c/blob/master/g4c_ac.h).
* We used Intel DPDK v1.7 for the EuroSys 2015 paper, but have now upgraded to v1.8.

## How to compile

We recommend to use Ubuntu 14.04 or newer.  
First you need to install some prerequisites.

* NVIDIA CUDA 6.0 or newer
  - We recommend to download the latest version of `.bin` package from [the NVIDIA website](https://developer.nvidia.com/cuda-downloads) instead of using system packages.
  - A small daemon is required to "pin" GPU's interrupts to specific cores.  
    See details in https://gist.github.com/3404967 .
  - Add `export PATH=$PATH:/usr/local/cuda/bin` to `/etc/profile` or similar places.
  - Add `/usr/local/cuda/lib64` to `/etc/ld.so.conf.d/cuda.conf` (create if not exists) and run `ldconfig`.
* g++ 4.8 or newer (the compiler must support C++11 standard.)
* Intel DPDK 1.8
  - Clone from git://dpdk.org/dpdk or download the release tarball.
  - Install the kernel header/source packages first.
  - Run `make install T=x86_64-native-linuxapp-gcc` in the checked-out directory.
* Python 3.4 or newer
  - `pip install snakemake`
  - `apt-get install libpython3-dev`
* Click configuration parser
  - https://github.com/leeopop/click-parser
  - Just checkout it in the home directory and compile in-place.
* If all set, then set the environment variable as follows:
  - `USE_CUDA=1`
  - `NBA_DPDK_PATH=/home/userid/dpdk/x86_64-native-linuxapp-gcc`
  - Then run `snakemake`
* If all is well, the executable is located in `bin/main`.

## Ongoing work

* Full datablock implementation
  - Base implementation is done but there remains room for performance optimization
* Decreasing GPU latency
  - Optimization for consequent offloadable modules: merge two separate offload-tasks into one
  - "Adaptive" batching
* OpenCL engine (primarily for Intel Xeon Phi)
* (Semi-)automated conversion of existing Click elements to NBA elements
