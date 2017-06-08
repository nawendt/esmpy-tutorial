# esmpy-tutorial
Basic tutorial for ESMPy Python package

ESMPy is the Python interface for the Earth System Modelling Framework (ESMF) regridding utility. You can find out more [here](https://www.earthsystemcog.org/projects/esmpy/). This repository contains a brief tutorial on how you can use this package to do some common regridding tasks. These examples use data from numerical weather prediction models as well as surface stations, all which are included. Take a look at the Jupyter Notebook and see how to use the package. Happy regridding!

Current examples available:
  * grid to grid (bilinear)
  * grid to grid (first-order conservative)
  * points to grid
  * grid to points
  * parallel grid to grid via MPI (bilinear)

### Running the parallel regrid example
Replace NPROCS with the number of processes you want to use and be sure `mpi4py` is installed.
```bash
mpirun -n NPROCS python esmpy_mpi_example.py
```

Questions about this tutorial? Have an example you would like to see? Contact me or submit an issue and I'll see what I can do to help. If you have specific questions about the ESMPy software or its development, you will be much better served by getting into contact with the actual ESMPy developers (I am _not_ one, just a happy user) at [esmf_support@list.woc.noaa.gov](mailto:esmf_support@list.woc.noaa.gov).
