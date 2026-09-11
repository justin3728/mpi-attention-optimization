"""Generate native-endian int32/float64 input and an independent FP64 reference."""
import argparse
import array
import math
import random
import struct
from pathlib import Path


def generate(path, m=17, n=19, dk=65, dv=33, seed=42):
    if min(m, n, dk, dv) <= 0:
        raise ValueError('Dimensions must be positive')
    rng = random.Random(seed)
    q = [rng.uniform(-1, 1) for _ in range(m * dk)]
    k = [rng.uniform(-1, 1) for _ in range(n * dk)]
    v = [rng.uniform(-1, 1) for _ in range(n * dv)]
    output = []
    for i in range(m):
        scores = [math.fsum(q[i * dk + t] * k[j * dk + t] for t in range(dk))
                  / math.sqrt(dk) for j in range(n)]
        maximum = max(scores)
        weights = [math.exp(s - maximum) for s in scores]
        total = math.fsum(weights)
        output.extend(math.fsum(weights[j] * v[j * dv + d] for j in range(n))
                      / total for d in range(dv))
    with Path(path).open('wb') as stream:
        stream.write(struct.pack('=4i', m, n, dk, dv))
        for values in (q, k, v, output):
            array.array('d', values).tofile(stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    for name, default in [('m', 17), ('n', 19), ('dk', 65), ('dv', 33), ('seed', 42)]:
        parser.add_argument('--' + name, type=int, default=default)
    args = vars(parser.parse_args())
    generate(args.pop('output'), **args)
