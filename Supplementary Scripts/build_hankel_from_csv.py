"""
Build Hankel matrices from a robot log CSV using PyDeePC utilities.

CSV expectations (14 columns total):
- diff_0 .. diff_6: positional gaps (used as outputs y)
- effort_0 .. effort_6: applied torques (used as inputs u)

The goal is to prepare DeePC data by forming the Hankel matrices and their
past/future partitions (Up, Uf, Yp, Yf) so they can be fed into the online
controller.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pydeepc.utils import (
    Data,
    create_hankel_matrix,
    low_rank_matrix_approximation,
    split_data,
)


DIFF_COLUMNS: List[str] = [f"diff_{idx}" for idx in range(7)]
EFFORT_COLUMNS: List[str] = [f"effort_{idx}" for idx in range(7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Hankel matrices (Hu, Hy, Up, Uf, Yp, Yf) from a CSV log.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=False,
        default=None,
        help="Path to the CSV containing diff_* and effort_* columns.",
    )
    parser.add_argument(
        "--tini",
        type=int,
        required=False,
        default=5,
        help="Number of samples for the initial window (Tini).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=False,
        default=30,
        help="Prediction horizon N.",
    )
    parser.add_argument(
        "--explained-variance",
        dest="explained_variance",
        type=float,
        default=None,
        help="Optional low-rank approximation target in (0,1].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .npz path. Defaults beside the CSV.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an internal consistency check with synthetic data.",
    )
    return parser.parse_args()


def load_csv_dataset(
    csv_path: Path, u_columns: Sequence[str], y_columns: Sequence[str]
) -> Data:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        missing = [col for col in (*u_columns, *y_columns) if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df[[*u_columns, *y_columns]].astype(np.float64)
        if df.isnull().values.any():
            raise ValueError("CSV contains NaN values; clean the data first.")
        u = df[u_columns].to_numpy(dtype=np.float64)
        y = df[y_columns].to_numpy(dtype=np.float64)
    except ImportError:
        # Fallback to the standard library CSV reader.
        u_rows, y_rows = [], []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row.")
            missing = [col for col in (*u_columns, *y_columns) if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            for idx, row in enumerate(reader):
                try:
                    u_rows.append([float(row[col]) for col in u_columns])
                    y_rows.append([float(row[col]) for col in y_columns])
                except KeyError as exc:
                    raise ValueError(f"Missing column {exc} in row {idx}") from exc
                except ValueError as exc:
                    raise ValueError(f"Non-numeric value in row {idx}") from exc

        if not u_rows:
            raise ValueError("CSV contained no data rows.")
        u = np.asarray(u_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)

    if u.shape[0] != y.shape[0]:
        raise ValueError(f"Input/output lengths differ: u={u.shape[0]}, y={y.shape[0]}")
    if u.shape[1] != len(u_columns) or y.shape[1] != len(y_columns):
        raise ValueError("Unexpected number of features in u or y.")

    return Data(u=u, y=y)


def build_hankel_components(
    data: Data, tini: int, horizon: int, explained_variance: Optional[float]
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    assert tini >= 1 and horizon >= 1, "tini and horizon must be positive"
    order = tini + horizon
    if data.u.shape[0] < order:
        raise ValueError(
            f"Not enough samples ({data.u.shape[0]}) for order {order}. "
            f"Need at least T >= tini + horizon."
        )

    Hu = create_hankel_matrix(data.u, order)
    Hy = create_hankel_matrix(data.y, order)

    if explained_variance is not None:
        Hu = low_rank_matrix_approximation(Hu, explained_var=explained_variance)
        Hy = low_rank_matrix_approximation(Hy, explained_var=explained_variance)

    Up, Uf, Yp, Yf = split_data(data, tini, horizon, explained_variance)

    rank_expected = min(Hu.shape[0], Hu.shape[1])
    rank_actual = int(np.linalg.matrix_rank(Hu))

    arrays = {"Hu": Hu, "Hy": Hy, "Up": Up, "Uf": Uf, "Yp": Yp, "Yf": Yf}
    rank_info = {"expected": rank_expected, "actual": rank_actual}
    return arrays, rank_info


def save_results(
    output_path: Path,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, object],
) -> None:
    output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays, meta=json.dumps(metadata, indent=2))
    print(f"Saved Hankel data to {output_path}")


def main(args: argparse.Namespace) -> None:
    if args.self_test:
        run_self_test()
        return

    if args.csv is None:
        raise SystemExit("Please provide --csv when not running --self-test.")

    tini, horizon, ev = args.tini, args.horizon, args.explained_variance
    data = load_csv_dataset(args.csv, EFFORT_COLUMNS, DIFF_COLUMNS)

    arrays, rank_info = build_hankel_components(data, tini, horizon, ev)
    order = tini + horizon

    output_path = (
        args.output
        if args.output is not None
        else args.csv.with_name(f"{args.csv.stem}_hankel_tini{tini}_N{horizon}.npz")
    )

    metadata = {
        "csv_path": str(args.csv),
        "T": int(data.u.shape[0]),
        "order": order,
        "tini": tini,
        "horizon": horizon,
        "explained_variance": ev,
        "u_columns": list(EFFORT_COLUMNS),
        "y_columns": list(DIFF_COLUMNS),
        "hu_rank": rank_info["actual"],
        "hu_rank_expected": rank_info["expected"],
        "hu_shape": tuple(arrays["Hu"].shape),
    }
    save_results(output_path, arrays, metadata)

    print("Summary")
    print(f"- Samples (T): {metadata['T']}")
    print(f"- Order (tini + horizon): {order}")
    print(f"- Hu shape: {arrays['Hu'].shape}, rank {rank_info['actual']} / {rank_info['expected']}")
    print(f"- Up shape: {arrays['Up'].shape}, Uf shape: {arrays['Uf'].shape}")
    print(f"- Yp shape: {arrays['Yp'].shape}, Yf shape: {arrays['Yf'].shape}")
    if ev is not None:
        print(f"- Low-rank approximation applied with explained_variance={ev}")
    if rank_info["actual"] < rank_info["expected"]:
        print("WARNING: Hu is rank-deficient; input data may not be persistently exciting.")


def run_self_test() -> None:
    """
    Generate synthetic data, write to a temporary CSV, and ensure shapes/ranks look sane.
    """
    import tempfile

    tini, horizon = 4, 6
    order = tini + horizon
    samples = 40
    time = np.arange(samples, dtype=np.float64)

    # Synthetic trajectories with mixed frequencies to avoid trivial rank loss.
    y_data = np.stack(
        [0.1 * (idx + 1) * np.sin(0.15 * time + 0.2 * idx) for idx in range(7)], axis=1
    )
    u_data = np.stack(
        [0.05 * (idx + 1) * np.cos(0.12 * time + 0.1 * idx) for idx in range(7)], axis=1
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=[*DIFF_COLUMNS, *EFFORT_COLUMNS])
        writer.writeheader()
        for row_idx in range(samples):
            row = {col: float(y_data[row_idx, idx]) for idx, col in enumerate(DIFF_COLUMNS)}
            row.update({col: float(u_data[row_idx, idx]) for idx, col in enumerate(EFFORT_COLUMNS)})
            writer.writerow(row)
        tmp_path = Path(tmp.name)

    try:
        data = load_csv_dataset(tmp_path, EFFORT_COLUMNS, DIFF_COLUMNS)
        arrays, rank_info = build_hankel_components(data, tini, horizon, explained_variance=None)
        assert arrays["Hu"].shape == (order * len(EFFORT_COLUMNS), samples - order + 1)
        assert arrays["Hy"].shape == (order * len(DIFF_COLUMNS), samples - order + 1)
        assert arrays["Up"].shape[0] == tini * len(EFFORT_COLUMNS)
        assert arrays["Uf"].shape[0] == horizon * len(EFFORT_COLUMNS)
        assert rank_info["actual"] >= min(arrays["Hu"].shape) - 1  # Allow tiny slack
        print("Self-test passed: synthetic dataset produced valid Hankel matrices.")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main(parse_args())
