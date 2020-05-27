from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'layer': 1, 'name': 'conv2'}
else:
    data = {'layer': 2, 'name': 'conv3', 'weights': np.zeros((2,3))}

data = comm.gather(data, root=0)

if rank == 0:
    print(data)
