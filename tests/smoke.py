"""Check SIMD tails, uneven/empty rank partitions, pipeline batches, and bad files."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_case import generate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--serial-only', action='store_true')
args = parser.parse_args()
suffix = '.exe' if sys.platform == 'win32' else ''
commands = [[str(ROOT / ('attention' + suffix))]]
if not args.serial_only:
    commands += [['mpiexec', '-n', str(ranks), str(ROOT / executable)]
                 for executable in ('attention-mpi', 'attention-mpi-scatter')
                 for ranks in (1, 2, 4)]


def run(command, path, success=True):
    proc = subprocess.run(command + [str(path)], capture_output=True, text=True, timeout=90)
    if (proc.returncode == 0) != success or (success and 'Correct!' not in proc.stdout):
        raise AssertionError(f'{command}: {proc.returncode}\n{proc.stdout}\n{proc.stderr}')


with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'case.bin'
    cases = [(1, 1, 1, 1), (17, 19, 65, 33), (7, 2, 17, 1), (1025, 5, 17, 19)]
    checks = 0
    for dims in cases:
        generate(path, *dims)
        for command in commands:
            run(command, path)
            checks += 1
    generate(path, 1, 1, 1, 1)
    valid = path.read_bytes()
    for broken in (b'', struct.pack('=4i', 0, 1, 1, 1), valid[:-1],
                   valid[:-8] + struct.pack('=d', float('nan')),
                   valid[:-8] + struct.pack('=d', 1000.0)):
        path.write_bytes(broken)
        for command in commands:
            run(command, path, success=False)
            checks += 1
print(f'Passed {checks} checks ({len(commands)} execution configurations).')
