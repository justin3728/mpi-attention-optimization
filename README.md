# Distributed Attention with MPI and AVX-512

A compact C implementation of scaled dot-product attention:
`softmax(Q K^T / sqrt(dk)) V`, optimized for multi-node CPU execution.

## Highlights

- Balanced K/V row partitions across MPI ranks, including uneven partitions.
- AVX-512 FMA dot products and weighted accumulation, loop unrolling, and masked tails.
- FP32 intermediate computation and communication with FP64 input/output.
- Online softmax and global maximum/sum reductions for stable normalization.
- 512-row query batches, nonblocking broadcasts/reductions, and ping-pong buffers.
- Adaptive K/V distribution: broadcast below 64 MiB; partitioned scatter otherwise.

## Reported performance

Original team experiments used **4 nodes, 16 MPI processes per node (64 total)**.

| Largest test case (`scale5`) | Speedup |
| --- | ---: |
| Versus the basic MPI baseline | **3.31x** |
| Versus the serial implementation | **7.49x** |

These are historical results from Team 12's HW3 report (Q2 and Q6), not new
measurements of this cleaned version. Original benchmark datasets and the basic
MPI baseline are not included; the supplied generator is for correctness checks.
Small inputs can be slower under MPI because communication dominates.

## Build and run

Requires a 64-bit Linux environment, GCC-compatible C compiler, MPI 3+ (such as
Open MPI), and an **AVX-512F/FMA-capable CPU on every rank**. Python 3 is needed
only to generate fixtures and run tests. The serial executable does not require
MPI or AVX-512. There is no SIMD fallback in the optimized executable.

```sh
make
python3 tools/generate_case.py sample.bin
./attention sample.bin
mpiexec -n 4 ./attention-mpi sample.bin
make test
```

To build only the serial reference: `make attention`.
To check it alone: `python3 tests/smoke.py --serial-only`.
For a cluster, select hosts and process placement using your MPI launcher or
scheduler. Network transport settings are left to the deployment environment.

Each executable reports correctness, maximum absolute error, and elapsed
microseconds. MPI timing is the maximum across ranks and includes conversion,
data distribution, computation, and result reductions inside `attention()`;
it excludes file I/O, validation, and MPI initialization.

## Source guide

| File | Purpose |
| --- | --- |
| `src/attention.c` | FP64 serial reference |
| `src/attention-mpi.c` | Distributed FP32 attention with AVX-512 kernels |
| `src/io.h` | Shared binary input, validation, and error handling |
| `tools/generate_case.py` | Deterministic fixtures with an independent FP64 reference |
| `tests/smoke.py` | SIMD tails, empty/uneven partitions, multiple batches, invalid files |

The binary format is four native-endian 32-bit integers (`m, n, dk, dv`), followed
by row-major FP64 arrays: Q (`m*dk`), K (`n*dk`), V (`n*dv`), and expected output
(`m*dv`). Generate and consume files on machines with the same byte order.
All dimensions must be positive and fit the implementation's integer count
limits. Input values must be suitable for FP32 computation: extreme finite FP64
values can overflow when converted or multiplied.

Verification rejects non-finite results and uses the original **0.02 absolute
error tolerance**. Converting the output back to FP64 does not restore precision
lost during FP32 computation. Communication overlap depends on MPI progress
and the target hardware; the max/sum collectives are immediately waited on.

## Provenance

Adapted from Team 12's course project, developed by 朱致伶, 李維哲 (Justin Lee),
and 江政諺. This portfolio edition retains the computational design, removes
assignment boilerplate and hard-coded cluster transport settings, shares the
I/O harness, and fixes validation and process exit handling.
