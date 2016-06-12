Network Balancing Act
=====================

**Network Balancing Act**, or **NBA**, is a packet processing framework based on Linux and commodity Intel x86 hardware.

Its goal is to offer both programmability and high performance by exploiting modern commodity hardware such as multi-queue-enabled 10 GbE network cards, multi-socket/multi-core CPUs, and computation accelerators like GPUs.
The programming interface resembles `the Click modular router <http://www.read.cs.ucla.edu/click/click>`_, which express packet processing functions as composable C++ classes (called *elements*).
Click has been a de-facto standard framework for software-based packet processors for years, but suffered from low performance.
In contrast, on Intel Sandy Bridge (E5-2670) dual-socket servers with eight 10 GbE (Intel X520-DA2) network cards, NBA saturates the hardware I/O capacity with light-weight workloads: 80 Gbps IPv4 forwarding throughput for large packets and 62 Gbps for minimum-sized packets.

Note that NBA covers per-packet processing only---it does not support flow-level processing yet.

.. _gsg:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started Guide

   gsg/sys_reqs
   gsg/build
   gsg/running
   gsg/docs

.. _user-docs:

.. toctree::
   :maxdepth: 2
   :caption: Using NBA

   user/system_config
   user/pipeline_config
   user/element
   user/offloadable_element

.. _elem-docs:

.. toctree::
   :maxdepth: 2
   :caption: Element Catalog

   elem/index

.. _dev-docs:

.. toctree::
   :maxdepth: 2
   :caption: Extending NBA

   dev/load_balancer
   dev/compute_device



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

