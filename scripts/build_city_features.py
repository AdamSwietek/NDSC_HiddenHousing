#!/usr/bin/env python3
"""Run the feature pipeline for a large polygon AOI in resumable tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.city_pipeline import load_polygon_aoi, run_tiled_feature_pipeline  # noqa: E402
from src.pipeline import ChangeJob, SnapshotJob  # noqa: E402


DEFAULT_SNAPSHOT_JOBS = (SnapshotJob(2017, 2017),)
DEFAULT_CHANGE_JOBS = (
    ChangeJob(2014, 2017, 2017),
    ChangeJob(2017, 2020, 2017),
    ChangeJob(2020, 2023, 2017),
)


def parse_snapshot_job(value):
    try:
        lariac_year, parcel_year = (int(part) for part in value.split(':'))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            'snapshot jobs must use LARIAC_YEAR:PARCEL_YEAR'
        ) from None
    return SnapshotJob(lariac_year, parcel_year)


def parse_change_job(value):
    try:
        start, end, reference_year = (int(part) for part in value.split(':'))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            'change jobs must use START_YEAR:END_YEAR:REFERENCE_PARCEL_YEAR'
        ) from None
    if end <= start:
        raise argparse.ArgumentTypeError('change END_YEAR must be later than START_YEAR')
    return ChangeJob(start, end, reference_year)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Build snapshot and change outputs for a polygon AOI using '
            'restartable, haloed tiles and an indexed assessor cache.'
        )
    )
    parser.add_argument(
        '--aoi-file',
        type=Path,
        default=PROJECT_ROOT / 'data/01_source/lacity_boundary.geojson',
    )
    parser.add_argument('--snapshot', action='append', type=parse_snapshot_job, default=[])
    parser.add_argument('--change', action='append', type=parse_change_job, default=[])
    parser.add_argument('--run-name', default='city_of_la')
    parser.add_argument(
        '--output-dir', type=Path, default=PROJECT_ROOT / 'data/03_processed'
    )
    parser.add_argument(
        '--work-dir', type=Path, default=PROJECT_ROOT / 'data/02_intermediate'
    )
    parser.add_argument('--assessor-cache', type=Path)
    parser.add_argument('--tile-size-miles', type=float, default=5.0)
    parser.add_argument('--tile-halo-feet', type=float, default=500.0)
    parser.add_argument('--alignment-sample-tiles', type=int, default=12)
    parser.add_argument('--snapshot-crs', default='EPSG:2229')
    parser.add_argument('--change-crs', default='EPSG:26911')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--no-change-geometries',
        action='store_true',
        help='Do not persist cleaned added and removed footprint layers.',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Snapshot assessor comparisons use the 2017 roll and 2017 geography.
    snapshots = args.snapshot or list(DEFAULT_SNAPSHOT_JOBS)
    # Every adjacent LARIAC epoch is compared on stable 2017 parcel geography.
    changes = args.change or list(DEFAULT_CHANGE_JOBS)
    aoi = load_polygon_aoi(args.aoi_file)
    output = run_tiled_feature_pipeline(
        aoi,
        snapshot_jobs=snapshots,
        change_jobs=changes,
        output_root=args.output_dir,
        work_root=args.work_dir,
        run_name=args.run_name,
        tile_size_miles=args.tile_size_miles,
        halo_feet=args.tile_halo_feet,
        snapshot_crs=args.snapshot_crs,
        change_crs=args.change_crs,
        assessor_cache_path=args.assessor_cache,
        alignment_sample_tiles=args.alignment_sample_tiles,
        write_change_geometries=not args.no_change_geometries,
        overwrite=args.overwrite,
    )
    print(f'Published citywide outputs: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
