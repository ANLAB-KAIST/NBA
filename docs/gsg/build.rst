Building NBA
============

We recommend to use Ubuntu 14.04 or newer.

Supported Platforms
-------------------

Currently NBA is only tested on Linux x86_64 3.10 or newer kernels,
and the Ubuntu 14.04 LTS distribution.

Step-by-step Guide for Ubuntu 14.04 LTS
---------------------------------------

**Software packages to install**

Ensure that you have a C/C++ compiler (e.g., g++ 4.8 or newer).
The compiler must support the C++11 standard.

Check out the latest DPDK source tree:

.. code-block:: console

   ~$ git clone git://dpdk.org/dpdk
   ~$ cd dpdk
   ~/dpdk$ make install T=x86_64-native-linuxapp-gcc

.. note::

   You need to install the kernel header/source packages first.

Install Python 3.4 on your system.
You may use the system package manager such as :code:`apt-get`.
In that case, ensure that you also have development package as well:

.. code-block:: console

  $ sudo apt-get install python3.4 libpython3.4-dev

Then install our Python dependencies:

.. code-block:: console

  $ pip3 install --user snakemake

.. note::

   We recommend using a separate Python environment contained inside the user directory.
   See `pyenv <https://github.com/yyuu/pyenv>`_ for more details.

Clone the project source code:

.. code-block:: console

   ~$ git clone https://github.com/anlab-kaist/NBA nba

Install our 3rd-party libraries, the Click configuration parser:

.. code-block:: console

   ~$ cd nba
   ~/nba$ git submodule init && git submodule update

.. note::

   It will be *automatically built* along with NBA when you first build NBA.


**Compilation**

* Set the environment variable as follows:

.. code-block:: console

  ~/nba$ export NBA_DPDK_PATH=/home/userid/dpdk/x86_64-native-linuxapp-gcc
  ~/nba$ snakemake

* If all is well, the executable is located in `bin/main`.

**Optional installation**

If you want to use GPU acceleration, install NVIDIA CUDA 7.0 or newer.
We recommend to download the latest version of :code:`.bin` package from `the NVIDIA website <https://developer.nvidia.com/cuda-downloads>`_ instead of using system packages.

.. note::

  A small daemon is required to "pin" GPU's interrupts to specific cores.
  See details in `our gist <https://gist.github.com/3404967>`_.

Make CUDA binaries accessible from your shell:

.. code-block:: console

  $ echo 'export PATH="$PATH:/usr/local/cuda/bin"' >> ~/.profile
  $ sudo sh -c 'echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf'
  $ sudo ldconfig

To run experiment scripts, install :code:`sysstat` package (or any package that offers :code:`mpstat` command).


Customizing Your Build
----------------------

Our build script offers a few configurable parameters as environment variables:

* :code:`NBA_DPDK_PATH`: specifies the path to Intel DPDK (required)
* :code:`NBA_RANDOM_PORT_ACCESS`: randomizes the RX queue scanning order for each worker thread (default: false)
* :code:`NBA_OPENSSL_PATH`: specifies the path of OpenSSL library (default: /usr)
* :code:`USE_CUDA`: activates NVIDIA CUDA support (default: true)
* :code:`USE_PHI`: activates Intel Xeon Phi support (default: false, not fully implemented yet)
* :code:`USE_NVPROF`: activates nvprof API calls to track GPU-related timings (default: false)
* :code:`USE_OPENSSL_EVP`: determines whether to use EVP API for OpenSSL that enables AES-NI support (default: true)
* :code:`NBA_NO_HUGE`: determines whether to use huge-pages (default: true)
* :code:`NBA_PMD`: determines what poll-mode driver to use (default: ixgbe)

※ Boolean variables are expressed as 1 or 0.
