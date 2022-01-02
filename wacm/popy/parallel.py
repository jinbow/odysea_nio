def start_mpi():
    from mpi4py import MPI

    comm=MPI.COMM_WORLD
    size=comm.Get_size()
    rank=comm.Get_rank()

    return rank,size,comm

def split(alist,thesize):
    nn=len(alist)

    dn=nn//(thesize-1)

    a=[]
    for i in range(thesize-1):
        a.append(alist[i*dn:i*dn+dn])

    a.append(alist[thesize*dn-dn:])
    return a

