# NBA (Network Balancing Act)

[![Join the chat at https://gitter.im/ANLAB-KAIST/NBA](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ANLAB-KAIST/NBA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

"A High-performance packet processing framework for heterogeneous processors," [EuroSys 2015 paper](http://an.kaist.ac.kr/~sbmoon/paper/intl-conf/2015-eurosys-nba.pdf)

## Notice for paper readers

* The IDS source code is not available to the public, as it contains a derivation from industry-transferred code from [Kargus](http://shader.kaist.edu/kargus/).
  - You could refer to other open-source code, such as [a GPU implementation of the Aho-Corasick algorithm in Snap](https://github.com/wbsun/g4c/blob/master/g4c_ac.h).
* We used Intel DPDK v1.7 for the EuroSys 2015 paper, but have now upgraded to v2.0+.

## Main Features

* 80-Gbps packet processing with modular programming interface similar to Click.
* GPU offloading to boost complex computations such as IPsec encryption
* Automatic CPU/GPU load balancing to maximize the overall throughput

## Documentation

[See the documentation online.](http://nba.readthedocs.org/en/latest/)

## Ongoing work

* Full datablock implementation
  - Base implementation is done but there remains room for performance optimization
* Decreasing GPU latency
  - Optimization for consequent offloadable modules: merge two separate offload-tasks into one
  - "Adaptive" batching
* OpenCL engine (primarily for Intel Xeon Phi)
* (Semi-)automated conversion of existing Click elements to NBA elements
