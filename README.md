# NBA (Network Balancing Act)
"A High-performance packet processing framework for heterogeneous processors," [EuroSys 2015 paper](http://an.kaist.ac.kr/~sbmoon/paper/intl-conf/2015-eurosys-nba.pdf)

## Disclaimer

* The IDS source code is not available to the public, as it contains a derivation from industry-transferred code from [Kargus](http://shader.kaist.edu/kargus/).
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
* `sysstat` package (or any package that offers `mpstat` command) to run experiment scripts
* If all set, then set the environment variable as follows:
  - `USE_CUDA=1`
  - `NBA_DPDK_PATH=/home/userid/dpdk/x86_64-native-linuxapp-gcc`
  - Then run `snakemake`
* If all is well, the executable is located in `bin/main`.

### Compile options

Our build script offers a few configurable parameters as environment variables:
* `NBA_DPDK_PATH`: specifies the path to Intel DPDK (required)
* `NBA_RANDOM_PORT_ACCESS`: randomizes the RX queue scanning order for each worker thread (default: false)
* `NBA_OPENSSL_PATH`: specifies the path of OpenSSL library (default: /usr)
* `USE_CUDA`: activates NVIDIA CUDA support (default: true)
* `USE_PHI`: activates Intel Xeon Phi support (default: false, not fully implemented yet)
* `USE_NVPROF`: activates nvprof API calls to track GPU-related timings (default: false)
* `USE_OPENSSL_EVP`: determines whether to use EVP API for OpenSSL that enables AES-NI support (default: true)
* `NBA_NO_HUGE`: determines whether to use huge-pages (default: true)
* `NBA_PMD`: determines what poll-mode driver to use (default: ixgbe)

※ Boolean variables are expressed as 1 or 0.

## How to run

Execute `bin/main` with DPDK EAL arguments and NBA arguments.  
For example,

```
$ sudo bin/main -cffff -n4 -- configs/rss.py configs/ipv4-router.click
```

## Ongoing work

* Full datablock implementation
  - Base implementation is done but there remains room for performance optimization
* Decreasing GPU latency
  - Optimization for consequent offloadable modules: merge two separate offload-tasks into one
  - "Adaptive" batching
* OpenCL engine (primarily for Intel Xeon Phi)
* (Semi-)automated conversion of existing Click elements to NBA elements
