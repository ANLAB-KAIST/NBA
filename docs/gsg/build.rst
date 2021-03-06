Building NBA
============

Supported Platforms
-------------------

Currently NBA is only tested on Linux x86_64 3.10 or newer kernels,
and the Ubuntu 14.04/16.04 LTS distribution.

Step-by-step Guide
------------------

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Ensure that you have a C/C++ compiler (e.g., g++ 4.8 or newer).
The compiler must support the C++11 standard.

Check out the latest DPDK source tree:

.. code-block:: console

   $ git clone git://dpdk.org/dpdk
   $ cd dpdk
   $ $EDITOR configs/common_linuxapp
   $ make install T=x86_64-native-linuxapp-gcc

See details on what you need to edit on :code:`configs/common_linuxapp`
at :ref:`DPDK configurations for network cards <dpdk-configs-for-nics>`.

.. note::

   You need to install the kernel header/source packages first.

Install library dependencies on your system:

.. code-block:: console

   $ sudo apt-get install libev-dev libssl-dev libpapi-dev

Install Python 3.5 on your system.
You may use the system package manager such as :code:`apt-get`.
In that case, ensure that you also have development package as well:

.. code-block:: console

   $ sudo apt-get install python3.5 libpython3.5-dev

Then install our Python dependencies:

.. code-block:: console

   $ pip3 install --user snakemake

.. note::

   We recommend using a separate Python environment contained inside the user directory.
   See `pyenv <https://github.com/yyuu/pyenv>`_ for more details.

Compilation
~~~~~~~~~~~

Clone the project source code:

.. code-block:: console

   $ git clone https://github.com/anlab-kaist/NBA nba

Install our 3rd-party libraries, the Click configuration parser:

.. code-block:: console

   $ cd nba
   $ git submodule init && git submodule update

It will be *automatically built* along with NBA when you first build NBA.

Set the environment variable as follows:

.. code-block:: console

   $ export NBA_DPDK_PATH=/home/userid/dpdk/x86_64-native-linuxapp-gcc
   $ export USE_CUDA=0  # for testing CPU-only version without CUDA installation

Finally, run:

.. code-block:: console

   $ snakemake -j

If all is well, the executable is located in :code:`bin/main`.


.. _dpdk-configs-for-nics:

DPDK Configs for Network Cards
------------------------------

Intel X520 Series (82599 chipset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You just need to bind the PCI addresses of network cards to igb_uio using
:code:`tools/dpdk_nic_bind.py` script provided by DPDK.

.. attention::

   When using ixgbe driver with *vectorized* PMD enabled, you should fix the IO
   batch size to be 32, whereas you may change the computation batch size as
   you want.

   In our experiements, the IO batch size 32 and the computation batch size 64 performs best.
   We have already set those as the default values. (see ``config/default.py``)

Mellanox ConnectX Series
~~~~~~~~~~~~~~~~~~~~~~~~

You need to install the OFED toolchain provided by Mellanox because DPDK's mlx4
poll-mode driver uses Mellanox's kernel Infiniband driver to control the
hardware and perform DMA.
We recommend to use version 3.0 or later, as these new versions have much
better performance and includes firmware updates.

To use mlx4_pmd on DPDK, turn on it inside DPDK's configuration:

.. code-block:: properties

   CONFIG_RTE_LIBRTE_MLX4_PMD=y

To increase throughputs, set the following in the same config:

.. code-block:: properties

   CONFIG_RTE_LIBRTE_MLX4_SGE_WR_N=1

For maximum throughputs, turn off the followings:

* blueflame [#blueflame]_: :code:`sudo ethtool --set-priv-flags ethXX blueflame off`
* rx/tx auto-negotiation for flow control: :code:`sudo ethtool -A ethXX rx off tx off`

Note that above settings must be done in packet generators as well.

.. warning::
   We recommend to turn off blueflame when loading the mlx4_core kernel module
   as module parameters, instead of using ethtool afterwards.

You do not need to explicitly bind the PCI addresses of Mellanox cards to
igb_uio because mlx4_pmd automatically detects them using the kernel driver.

To use mlx4 in NBA, set the following environment variable and rebuild:

.. code-block:: console

   $ export NBA_PMD=mlx4
   $ snakemake clean && snakemake -j

.. [#blueflame]
   "blueflame" is a Mellanox-specific feature that uses PCI BAR for tranferring
   descriptors of small packets instead of using DMA on RX/TX rings.  It is
   known to have lower latency, but causes throughput degradation with NBA.

Optional Installations
----------------------

NVIDIA CUDA
~~~~~~~~~~~

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

To use CUDA in NBA, do:

.. code-block:: console

   $ export USE_CUDA=1
   $ snakemake clean && snakemake -j

Intel Xeon Phi
~~~~~~~~~~~~~~

If you want to use Xeon Phi acceleration, install the latest Intel MPSS (many-core platform software stack) by visiting `the official website <https://software.intel.com/en-us/articles/intel-manycore-platform-software-stack-mpss>`_.

CPU statistics
~~~~~~~~~~~~~~

To run experiment scripts, install :code:`sysstat` package (or any package that offers :code:`mpstat` command).


Customizing Your Build
----------------------

Our build script offers a few configurable parameters as environment variables:

* :envvar:`NBA_DPDK_PATH`: specifies the path to Intel DPDK (required)
* :envvar:`NBA_RANDOM_PORT_ACCESS`: randomizes the RX queue scanning order for each worker thread (default: :code:`false`)
* :envvar:`NBA_OPENSSL_PATH`: specifies the path of OpenSSL library (default: :code:`/usr`)
* :envvar:`DEBUG`: build without compiler optimization (default: 0)
* :envvar:`USE_CUDA`: activates NVIDIA CUDA support (default: 1)
* :envvar:`USE_KNAPP`: activates Knapp-based Intel Xeon Phi support (default: 0)
* :envvar:`USE_PHI`: activates OpenCL-based Intel Xeon Phi support (default: 0, not implemented)
* :envvar:`USE_NVPROF`: activates nvprof API calls to track GPU-related timings (default: 0)
* :envvar:`USE_OPENSSL_EVP`: determines whether to use EVP API for OpenSSL that enables AES-NI support (default: 1)
* :envvar:`NBA_NO_HUGE`: determines whether to use huge-pages (default: 1)
* :envvar:`NBA_PMD`: determines what poll-mode driver to use (default: :code:`ixgbe`)

.. note::

   1 means true and 0 means false for boolean options.
