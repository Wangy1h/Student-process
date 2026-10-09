"""Compare regenerated simulation arrays with the saved manuscript outputs."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('regenerated', type=Path, help='Regenerated simulation_arrays.npz')
    parser.add_argument('--reference', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data' / 'simulation_arrays.npz')
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as expected, \
            np.load(args.regenerated, allow_pickle=False) as actual:
        if set(expected.files) != set(actual.files):
            raise ValueError('Archive array names differ')
        for name in expected.files:
            x, y = expected[name], actual[name]
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y):
                raise ValueError(f'Array differs: {name}')
        print(json.dumps({'array_count': len(expected.files),
                          'array_names_shapes_dtypes_values_identical': True}, indent=2))


if __name__ == '__main__':
    main()
