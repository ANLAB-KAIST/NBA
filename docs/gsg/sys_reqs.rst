System Requirements
===================

Supported Hardware
------------------

NBA can run on most Intel servers with supported NICs,
but we recommend Sandy Bridge or newer generations of Intel CPUs
since integrated memory controller and integrated PCIe controller is a
important performance booster.

For GPU acceleration, the current version only supports NVIDIA CUDA GPUs,
either desktop class (GeForce) or server class (Tesla) models.

BIOS Settings
-------------

* If available, turn off Intel EIST technology in BIOS.

* You need to fix the clock speed of individual CPU cores on Haswell systems
  (Xeon E5-XXXXv3) for accurate timing control and performance measurements.
