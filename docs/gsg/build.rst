Building NBA
============

We recommend to use Ubuntu 14.04 or newer.  

Supported Platforms
-------------------

Step-by-step Guide for Ubuntu 14.04 LTS
---------------------------------------

** Package and software to install **

* NVIDIA CUDA 7.0 or newer
  - We recommend to download the latest version of `.bin` package from `the NVIDIA website<https://developer.nvidia.com/cuda-downloads>`_ instead of using system packages.

.. note::

  A small daemon is required to "pin" GPU's interrupts to specific cores.  
  See details in `our gist <https://gist.github.com/3404967>`_.

  - Add `export PATH=$PATH:/usr/local/cuda/bin` to `/etc/profile` or similar places.
  - Add `/usr/local/cuda/lib64` to `/etc/ld.so.conf.d/cuda.conf` (create if not exists) and run `ldconfig`.

* g++ 4.8 or newer (the compiler must support C++11 standard.)

* Intel DPDK 1.8
.. code-block:: console

   ~$ git clone git://dpdk.org/dpdk
   ~$ cd dpdk
   ~/dpdk$ make install T=x86_64-native-linuxapp-gcc

.. note::

   You need to install the kernel header/source packages first.

* Python 3.4 or newer

.. code-block:: console

  pip install snakemake
  apt-get install libpython3-dev

.. note::

   We recommend using a separate Python environment contained inside the user directory.
   See pyenv for more details.

* Click configuration parser
  - Just run `git submodule init && git submodule update`
  - It will be *automatically built* along with NBA when you first build NBA.

* `sysstat` package (or any package that offers `mpstat` command) to run experiment scripts

** Compilation **

* Set the environment variable as follows:

.. code-block:: console

  $ export NBA_DPDK_PATH=/home/userid/dpdk/x86_64-native-linuxapp-gcc
  $ snakemake

* If all is well, the executable is located in `bin/main`.

Customizing Your Build
----------------------

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
