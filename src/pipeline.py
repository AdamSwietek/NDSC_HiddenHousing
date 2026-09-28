"""Orchestration for snapshot and change feature construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import geopandas as gpd
import pandas as pd

from .change_metrics import (
    DEFAULT_CHANGE_CRS,
    calculate_change_metrics,
    prepare_epoch_footprints,
)
from .io import load_assessor_columns, load_lariac, load_parcel_polygons
from .metrics import build_snapshot_metrics


DEFAULT_SNAPSHOT_CRS = 'EPSG:2229'
DEFAULT_ASSESSOR_COLUMNS = (
    'SQFTmain',
    'Units',
    'Bedrooms',
    'Bathrooms',
    'YearBuilt',
)
PIPELINE_LARIAC_COLUMNS = (
    'building_id',
    'height',
    'elevation',
    'status',
    'source',
    'capture_date',
    'old_building_id',
)


@dataclass(frozen=True)
class SnapshotJob:
    """One LARIAC snapshot paired with its parcel geography."""

    lariac_year: int
    parcel_year: int


@dataclass(frozen=True)
class ChangeJob:
    """One epoch comparison on a stable reference parcel geography."""

    start_year: int
    end_year: int
    reference_parcel_year: int


@dataclass
class FeaturePipelineResult:
    """In-memory outputs from one feature-pipeline run."""

    building_metrics: dict[int, gpd.GeoDataFrame] = field(default_factory=dict)
    parcel_metrics: dict[int, gpd.GeoDataFrame] = field(default_factory=dict)
    assessor_ain_metrics: dict[int, pd.DataFrame] = field(default_factory=dict)
    change_results: dict[tuple[int, int], dict] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def row_summary(self) -> dict[str, int]:
        summary = {}
        for year, frame in self.building_metrics.items():
            summary[f'building_metrics_{year}'] = int(len(frame))
        for year, frame in self.parcel_metrics.items():
            summary[f'parcel_metrics_{year}'] = int(len(frame))
        for year, frame in self.assessor_ain_metrics.items():
            summary[f'assessor_ain_metrics_{year}'] = int(len(frame))
        for pair, result in self.change_results.items():
            summary[f'change_metrics_{pair[0]}_{pair[1]}'] = int(
                len(result['metrics'])
            )
        return summary


def run_feature_pipeline(
    aoi,
    *,
    snapshot_jobs: Sequence[SnapshotJob] = (),
    change_jobs: Sequence[ChangeJob] = (),
    snapshot_crs=DEFAULT_SNAPSHOT_CRS,
    change_crs=DEFAULT_CHANGE_CRS,
    assessor_columns: Sequence[str] = DEFAULT_ASSESSOR_COLUMNS,
    assessor_db_filepath=None,
    repair_boundary_artifacts=True,
    repair_options=None,
    change_alignments=None,
    retain_change_debug_geometries=True,
) -> FeaturePipelineResult:
    """Load each source once and build all requested in-memory feature sets.

    Persistence is intentionally outside this function. Task 7 can define a
    stable output contract without coupling file formats to source loading and
    feature construction.
    """
    snapshots = _normalize_snapshot_jobs(snapshot_jobs)
    changes = _normalize_change_jobs(change_jobs)
    if not snapshots and not changes:
        raise ValueError('Provide at least one snapshot or change job')

    lariac_years = {
        job.lariac_year for job in snapshots
    } | {
        year for job in changes for year in (job.start_year, job.end_year)
    }
    parcel_years = {
        job.parcel_year for job in snapshots
    } | {
        job.reference_parcel_year for job in changes
    }

    buildings = {
        year: load_lariac(
            year,
            aoi,
            columns=PIPELINE_LARIAC_COLUMNS,
            target_crs=snapshot_crs,
        )
        for year in sorted(lariac_years)
    }
    parcels = {
        year: load_parcel_polygons(
            year,
            aoi,
            columns=['ASSETID', 'PIND', 'PIN'],
            target_crs=snapshot_crs,
            include_missing_ain=True,
        )
        for year in sorted(parcel_years)
    }

    result = FeaturePipelineResult(
        diagnostics={
            'sources': {
                'lariac_rows': {
                    year: int(len(frame)) for year, frame in buildings.items()
                },
                'parcel_rows': {
                    year: int(len(frame)) for year, frame in parcels.items()
                },
                'parcel_loading': {
                    year: frame.attrs.get('loading_diagnostics', {})
                    for year, frame in parcels.items()
                },
            },
            'snapshots': {},
            'changes': {},
        }
    )

    for job in snapshots:
        assessor = load_assessor_for_sites(
            job.lariac_year,
            assessor_columns,
            aoi,
            parcels[job.parcel_year],
            db_filepath=assessor_db_filepath,
        )
        building_output, parcel_output = build_snapshot_metrics(
            buildings[job.lariac_year],
            parcels[job.parcel_year],
            assessor,
            job.lariac_year,
            target_crs=snapshot_crs,
            repair_boundary_artifacts=repair_boundary_artifacts,
            repair_options=repair_options,
        )
        result.building_metrics[job.lariac_year] = building_output
        result.parcel_metrics[job.lariac_year] = parcel_output
        result.assessor_ain_metrics[job.lariac_year] = parcel_output.attrs.get(
            'site_ain_bridge', pd.DataFrame()
        )
        result.diagnostics['snapshots'][job.lariac_year] = {
            'parcel_year': job.parcel_year,
            'matching': building_output.attrs.get('matching_diagnostics', {}),
            'geometry_cleaning': building_output.attrs.get(
                'geometry_cleaning_diagnostics', {}
            ),
            'boundary_repair': building_output.attrs.get(
                'boundary_repair_diagnostics', {}
            ),
        }

    prepared_changes = {}
    for job in changes:
        alignment_override = (change_alignments or {}).get(
            (job.start_year, job.end_year)
        )
        footprints = {}
        for year in (job.start_year, job.end_year):
            prepared_key = (year, job.reference_parcel_year)
            if prepared_key not in prepared_changes:
                prepared_changes[prepared_key] = prepare_epoch_footprints(
                    buildings[year],
                    parcels[job.reference_parcel_year],
                    year,
                    target_crs=change_crs,
                    repair_boundary_artifacts=repair_boundary_artifacts,
                    repair_options=repair_options,
                )
            footprints[year] = prepared_changes[prepared_key]
        change_output = calculate_change_metrics(
            footprints[job.start_year],
            footprints[job.end_year],
            parcels[job.reference_parcel_year],
            store_geometries=True,
            alignment_override=alignment_override,
        )
        change_output['footprints'] = footprints
        key = (job.start_year, job.end_year)
        result.change_results[key] = change_output
        start = change_output['footprints'][job.start_year]
        end = change_output['footprints'][job.end_year]
        result.diagnostics['changes'][f'{job.start_year}_{job.end_year}'] = {
            'reference_parcel_year': job.reference_parcel_year,
            'comparison': change_output['diagnostics'],
            'assignment': {
                job.start_year: start.attrs.get('assignment_diagnostics', {}),
                job.end_year: end.attrs.get('assignment_diagnostics', {}),
            },
            'boundary_repair': {
                job.start_year: start.attrs.get(
                    'boundary_repair_diagnostics', {}
                ),
                job.end_year: end.attrs.get(
                    'boundary_repair_diagnostics', {}
                ),
            },
        }
        if not retain_change_debug_geometries:
            for debug_key in (
                'raw_added', 'raw_removed', 'filtered_added',
                'filtered_removed', 'aligned_end',
            ):
                change_output.pop(debug_key, None)

    result.diagnostics['rows'] = result.row_summary()
    return result


def load_assessor_for_sites(
    roll_year, columns, aoi, parcels, *, db_filepath=None
):
    """Load the union of exact-AIN and AOI-center assessor records."""
    requested = list(dict.fromkeys([*columns, 'CENTER_LON', 'CENTER_LAT']))
    known_ains = parcels['AIN'].dropna().drop_duplicates()
    parts = []
    if not known_ains.empty:
        parts.append(
            load_assessor_columns(
                roll_year,
                requested,
                ains=known_ains,
                db_filepath=db_filepath,
            )
        )
    parts.append(
        load_assessor_columns(
            roll_year, requested, aoi=aoi, db_filepath=db_filepath
        )
    )
    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined = combined.drop_duplicates('AIN', keep='last').reset_index(drop=True)
    if {'CENTER_LON', 'CENTER_LAT'}.issubset(combined.columns):
        combined = combined.drop(columns='geometry', errors='ignore')
        combined = gpd.GeoDataFrame(
            combined,
            geometry=gpd.points_from_xy(
                combined['CENTER_LON'], combined['CENTER_LAT']
            ),
            crs='EPSG:4326',
        )
    return combined


def _normalize_snapshot_jobs(jobs):
    normalized = [
        job if isinstance(job, SnapshotJob) else SnapshotJob(*job)
        for job in jobs
    ]
    years = [int(job.lariac_year) for job in normalized]
    if len(years) != len(set(years)):
        raise ValueError('Each snapshot LARIAC year may appear only once')
    return [SnapshotJob(int(job.lariac_year), int(job.parcel_year)) for job in normalized]


def _normalize_change_jobs(jobs):
    normalized = [
        job if isinstance(job, ChangeJob) else ChangeJob(*job)
        for job in jobs
    ]
    pairs = []
    output = []
    for job in normalized:
        start = int(job.start_year)
        end = int(job.end_year)
        if end <= start:
            raise ValueError('Change end year must be later than start year')
        pair = (start, end)
        if pair in pairs:
            raise ValueError(f'Duplicate change pair {start}:{end}')
        pairs.append(pair)
        output.append(ChangeJob(start, end, int(job.reference_parcel_year)))
    return output
