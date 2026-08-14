from numbers import Integral

import numpy as np

from scheduling_environment.job import Job
from scheduling_environment.jobShop import JobShop
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation


def permute_rows(x, random_state=None):
    """Randomly permute every row in ``x`` independently."""
    random_state = np.random if random_state is None else random_state
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = random_state.random_sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def l2d_instance_to_job_shop(instance, instance_name="generated_l2d_instance"):
    """Convert an L2D ``(processing_times, machines)`` tuple to ``JobShop``.

    L2D represents machines with one-based IDs.  The benchmark environment uses
    zero-based IDs, so the conversion subtracts one from every machine ID.

    Parameters
    ----------
    instance : tuple[numpy.ndarray, numpy.ndarray]
        Two equally shaped ``(number_of_jobs, number_of_machines)`` arrays. The
        first contains processing times and the second contains one-based
        machine IDs.
    instance_name : str, optional
        Label stored on the resulting environment.

    Returns
    -------
    JobShop
        An initialized, unscheduled benchmark environment that can be passed to
        a solver and then to the precedence or Gantt-chart visualizer.
    """
    try:
        number_of_arrays = len(instance)
    except TypeError as error:
        raise ValueError("instance must be a (processing_times, machines) pair") from error
    if number_of_arrays != 2:
        raise ValueError("instance must be a (processing_times, machines) pair")

    processing_times = np.asarray(instance[0])
    machines = np.asarray(instance[1])
    if processing_times.ndim != 2 or machines.ndim != 2:
        raise ValueError("processing_times and machines must both be 2-D arrays")
    if processing_times.shape != machines.shape:
        raise ValueError("processing_times and machines must have the same shape")
    if 0 in processing_times.shape:
        raise ValueError("an instance must contain at least one job and one machine")
    if not np.all(np.isfinite(processing_times)) or np.any(processing_times <= 0):
        raise ValueError("processing times must be finite and strictly positive")
    if not np.issubdtype(machines.dtype, np.integer):
        raise ValueError("machine IDs must be integers")

    n_j, n_m = processing_times.shape
    expected_machines = np.arange(1, n_m + 1)
    if not all(np.array_equal(np.sort(row), expected_machines) for row in machines):
        raise ValueError(f"each job must use every L2D machine ID from 1 to {n_m} exactly once")

    job_shop = JobShop()
    job_shop.set_instance_name(instance_name)
    job_shop.set_nr_of_jobs(n_j)
    job_shop.set_nr_of_machines(n_m)
    precedence_relations = {}

    operation_id = 0
    for job_id in range(n_j):
        job = Job(job_id)
        previous_operation = None

        for operation_index in range(n_m):
            operation = Operation(job, job_id, operation_id)
            machine_id = int(machines[job_id, operation_index]) - 1
            duration = processing_times[job_id, operation_index].item()
            operation.add_operation_option(machine_id, duration)
            operation.add_predecessors([] if previous_operation is None else [previous_operation])

            precedence_relations[operation_id] = list(operation.predecessors)
            job.add_operation(operation)
            job_shop.add_operation(operation)
            previous_operation = operation
            operation_id += 1

        job_shop.add_job(job)

    job_shop.add_precedence_relations_operations(precedence_relations)
    job_shop.add_sequence_dependent_setup_times(
        [[[0] * operation_id for _ in range(operation_id)] for _ in range(n_m)]
    )
    for machine_id in range(n_m):
        job_shop.add_machine(Machine(machine_id))

    job_shop.reset()
    return job_shop


def uniform_instance_generator(
        n_j,
        n_m,
        low,
        high,
        *,
        seed=None,
        return_job_shop=False,
        instance_name=None):
    """Generate a uniformly distributed classical JSSP instance for L2D.

    By default, this keeps the original L2D API and returns only
    ``(processing_times, machines)``. Set ``return_job_shop=True`` to also get a
    matching benchmark environment for visualization::

        l2d_instance, job_shop = uniform_instance_generator(
            6, 6, 1, 100, seed=200, return_job_shop=True
        )

    ``high`` is exclusive, as in :func:`numpy.random.randint`. Supplying
    ``seed`` makes this call reproducible without changing NumPy's global random
    state. Leaving it unset preserves the training script's seeded global RNG.
    """
    for value, name in ((n_j, "n_j"), (n_m, "n_m"), (low, "low"), (high, "high")):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if n_j <= 0 or n_m <= 0:
        raise ValueError("n_j and n_m must be greater than zero")
    if low <= 0:
        raise ValueError("low must be greater than zero")
    if high <= low:
        raise ValueError("high must be greater than low")

    random_state = np.random if seed is None else np.random.RandomState(seed)
    times = random_state.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines, random_state=random_state)
    l2d_instance = (times, machines)

    if not return_job_shop:
        return l2d_instance

    name = instance_name or f"uniform_{n_j}x{n_m}_seed_{seed}"
    return l2d_instance, l2d_instance_to_job_shop(l2d_instance, instance_name=name)


def override(fn):
    """
    override decorator
    """
    return fn


if __name__ == "__main__":
    # Set parameters
    j = 20              # Number of jobs
    m = 10              # Number of machines
    l = 1               # Minimum processing time
    h = 99              # Maximum processing time
    batch_size = 100    # nr of instances to generate
    seed = 201          # Random seed

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate data for batch_size number of instances
    data = np.array([uniform_instance_generator(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    print(f"Generated data shape: {data.shape}")
    np.save(f'generated_data/test_generatedData{j}_{m}_Seed{seed}.npy', data)
