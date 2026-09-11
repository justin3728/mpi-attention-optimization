CC = cc
MPICC = mpicc
CFLAGS = -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra
SIMD_FLAGS = -mavx512f -mfma

.PHONY: all clean test
all: attention attention-mpi

attention: src/attention.c src/io.h
	$(CC) $(CFLAGS) $< -lm -o $@

attention-mpi: src/attention-mpi.c src/io.h
	$(MPICC) $(CFLAGS) $(SIMD_FLAGS) $< -lm -o $@

# Exercise the Scatterv branch on small generated fixtures.
attention-mpi-scatter: src/attention-mpi.c src/io.h
	$(MPICC) $(CFLAGS) $(SIMD_FLAGS) -DKV_BCAST_THRESHOLD=0 $< -lm -o $@

test: all attention-mpi-scatter
	python3 tests/smoke.py

clean:
	rm -f attention attention-mpi attention-mpi-scatter
