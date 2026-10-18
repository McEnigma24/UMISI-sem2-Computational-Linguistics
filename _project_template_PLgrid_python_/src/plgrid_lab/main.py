import os


def _slurm_cpu_summary() -> str:
    ntasks = os.environ.get("SLURM_NTASKS", "?")
    cpt = os.environ.get("SLURM_CPUS_PER_TASK", "?")
    omp = os.environ.get("OMP_NUM_THREADS", "?")
    try:
        total = int(ntasks) * int(cpt)
    except ValueError:
        total = "?"
    return (
        f"SLURM_NTASKS={ntasks} SLURM_CPUS_PER_TASK={cpt} "
        f"OMP_NUM_THREADS={omp} (łącznie rdzeni CPU ≈ {total})"
    )


def main() -> None:
    if "LOCAL_RANK" in os.environ:
        import torch

        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(
            f"torchrun local_rank={local_rank} "
            f"world_size={os.environ.get('WORLD_SIZE')} "
            f"device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
        )
        if local_rank == 0:
            print(_slurm_cpu_summary())
    else:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Hello, World! I am rank {rank} of {size}")
        comm.Barrier()
        print(f"Hello, World! I am rank {rank} of {size} after barrier")
        if rank == 0:
            print(_slurm_cpu_summary())


if __name__ == "__main__":
    main()
