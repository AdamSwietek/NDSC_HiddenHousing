#!/usr/bin/env python3
"""Run snapshot and change feature construction for a bounded AOI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (  # noqa: E402
    ChangeJob,
    SnapshotJob,
    run_feature_pipeline,
)
from src.outputs import write_feature_outputs  # noqa: E402


def parse_snapshot_job(value: str) -> SnapshotJob:
    try:
        lariac_year, parcel_year = (int(part) for part in value.split(':'))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            'snapshot jobs must use LARIAC_YEAR:PARCEL_YEAR'
        ) from None
    return SnapshotJob(lariac_year, parcel_year)


def parse_change_job(value: str) -> ChangeJob:
    try:
        start, end, parcel_year = (int(part) for part in value.split(':'))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            'change jobs must use START_YEAR:END_YEAR:REFERENCE_PARCEL_YEAR'
        ) from None
    if end <= start:
        raise argparse.ArgumentTypeError('change END_YEAR must be later than START_YEAR')
    return ChangeJob(start, end, parcel_year)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Load each LARIAC/parcel vintage once and build snapshot and '
            'change features for a bounded AOI.'
        )
    )
    aoi_group = parser.add_mutually_exclusive_group(required=True)
    aoi_group.add_argument(
        '--aoi',
        nargs=4,
        type=float,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help='WGS84 bounding box used to restrict every source load.',
    )
    aoi_group.add_argument(
        '--aoi-file',
        type=Path,
        help='Polygon dataset used as the exact AOI (all features are unioned).',
    )
    parser.add_argument(
        '--snapshot',
        action='append',
        type=parse_snapshot_job,
        default=[],
        metavar='LARIAC_YEAR:PARCEL_YEAR',
        help='Snapshot job; repeat for multiple years.',
    )
    parser.add_argument(
        '--change',
        action='append',
        type=parse_change_job,
        default=[],
        metavar='START:END:REFERENCE_PARCEL_YEAR',
        help='Change job; repeat for multiple epoch pairs.',
    )
    parser.add_argument('--snapshot-crs', default='EPSG:2229')
    parser.add_argument('--change-crs', default='EPSG:26911')
    parser.add_argument(
        '--no-boundary-repair',
        action='store_true',
        help='Disable source-level parcel-boundary fragment reconstruction.',
    )
    parser.add_argument(
        '--summary-json',
        type=Path,
        help='Optional path for row counts and run diagnostics only.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'data/03_processed',
        help='Parent directory for persistent run folders.',
    )
    parser.add_argument(
        '--run-name',
        help='Write persistent outputs under OUTPUT_DIR/RUN_NAME.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing run with the same name.',
    )
    parser.add_argument(
        '--write-change-geometries',
        action='store_true',
        help='Also write cleaned added and removed change-geometry layers.',
    )
    parser.add_argument(
        '--assessor-db',
        type=Path,
        help='Optional assessor SQLite database, including a prepared cache.',
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    aoi, aoi_bounds = load_aoi(args)
    snapshots = list(args.snapshot)
    changes = list(args.change)
    if not snapshots and not changes:
        snapshots = [SnapshotJob(2020, 2021)]
        changes = [ChangeJob(2017, 2020, 2017)]

    result = run_feature_pipeline(
        aoi,
        snapshot_jobs=snapshots,
        change_jobs=changes,
        snapshot_crs=args.snapshot_crs,
        change_crs=args.change_crs,
        assessor_db_filepath=args.assessor_db,
        repair_boundary_artifacts=not args.no_boundary_repair,
        retain_change_debug_geometries=False,
    )
    parameters = {
        'snapshot_jobs': [
            {
                'lariac_year': job.lariac_year,
                'parcel_year': job.parcel_year,
            }
            for job in snapshots
        ],
        'change_jobs': [
            {
                'start_year': job.start_year,
                'end_year': job.end_year,
                'reference_parcel_year': job.reference_parcel_year,
            }
            for job in changes
        ],
        'snapshot_crs': args.snapshot_crs,
        'change_crs': args.change_crs,
        'repair_boundary_artifacts': not args.no_boundary_repair,
        'write_change_geometries': bool(args.write_change_geometries),
        'aoi_file': str(args.aoi_file) if args.aoi_file else None,
        'assessor_db': str(args.assessor_db) if args.assessor_db else None,
    }
    output_path = None
    if args.run_name:
        output_path = write_feature_outputs(
            result,
            args.output_dir,
            args.run_name,
            aoi_wgs84=aoi_bounds,
            parameters=parameters,
            overwrite=args.overwrite,
            write_change_geometries=args.write_change_geometries,
        )
    summary = {
        'aoi_wgs84': list(aoi_bounds),
        'rows': result.row_summary(),
        'diagnostics': result.diagnostics,
        'output_path': str(output_path) if output_path else None,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True, default=str)
    print(rendered)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered + '\n')
        print(f'Wrote diagnostics: {args.summary_json}', file=sys.stderr)
    return 0


def load_aoi(args):
    """Return an exact WGS84 AOI and its manifest bounding box."""
    if args.aoi_file is None:
        bounds = tuple(float(value) for value in args.aoi)
        return bounds, bounds
    if not args.aoi_file.exists():
        raise FileNotFoundError(args.aoi_file)
    frame = gpd.read_file(args.aoi_file)
    if frame.crs is None:
        raise ValueError(f'AOI file has no CRS: {args.aoi_file}')
    present = frame.geometry.notna() & ~frame.geometry.is_empty
    frame = frame.loc[present].to_crs('EPSG:4326')
    if frame.empty:
        raise ValueError(f'AOI file has no usable geometry: {args.aoi_file}')
    geometry = frame.geometry.make_valid().union_all()
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f'AOI file union is invalid: {args.aoi_file}')
    aoi = gpd.GeoDataFrame({'aoi': [args.aoi_file.stem]}, geometry=[geometry], crs='EPSG:4326')
    return aoi, tuple(float(value) for value in aoi.total_bounds)


if __name__ == '__main__':
    raise SystemExit(main())
