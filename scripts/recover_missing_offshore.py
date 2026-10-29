"""Recover only missing offshore model windows; leave existing forecasts intact."""
import argparse
from datetime import date
from pathlib import Path
import pandas as pd

from rt_forecast_dashboard.config import zones, features
from rt_forecast_dashboard.data.entsoe_client import EntsoeForecastClient
from rt_forecast_dashboard.models.registry import enabled_models_for_target
from rt_forecast_dashboard.pipeline import run_daily as daily
from rt_forecast_dashboard.storage import ForecastStore
from rt_forecast_dashboard.time_utils import brussels_cutoff_timestamp, forecast_horizon_hours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--end', type=date.fromisoformat, required=True)
    parser.add_argument('--zones', default='BE,FR')
    args = parser.parse_args()
    if args.start > args.end:
        raise ValueError('Start must precede end')
    selected = {z: zones()[z] for z in args.zones.split(',')}
    target = 'wind_offshore'
    model_keys = set(enabled_models_for_target(target)) - {'tabpfn_online'}
    store = ForecastStore()
    client = EntsoeForecastClient()
    if not client.api_key:
        raise RuntimeError('Recovery requires ENTSOE_API_KEY to repair historical actual context')
    original_fetch = EntsoeForecastClient.fetch_direct_actual_context
    context_hours = int(features()[target]['context_hours'])
    cache = {}
    # One broad normalized query per zone avoids repeating ninety-day API calls
    # for every missing date and model. The per-date view excludes future actuals.
    def cached_fetch(self, *, run_date, zone_code, target, context_hours):
        key = (zone_code, target)
        if key not in cache:
            broad_hours = context_hours + int((brussels_cutoff_timestamp(args.end) - brussels_cutoff_timestamp(args.start)) / pd.Timedelta(hours=1))
            cache[key] = original_fetch(self, run_date=args.end, zone_code=zone_code,
                                        target=target, context_hours=broad_hours)
        cutoff = brussels_cutoff_timestamp(run_date)
        first = cutoff - pd.Timedelta(hours=context_hours + 72)
        frame = cache[key]
        return frame[frame.timestamp.ge(first) & frame.timestamp.lt(cutoff)].copy()
    EntsoeForecastClient.fetch_direct_actual_context = cached_fetch
    target_config = features()[target]
    daily.features = lambda: {target: target_config}
    report = []
    for day in pd.date_range(args.start, args.end, freq='D'):
        run_date = day.date()
        for zone, config in selected.items():
            path = store.forecast_path(run_date.isoformat())
            saved = pd.read_csv(path) if path.exists() else pd.DataFrame()
            complete = set()
            if not saved.empty:
                part = saved[saved.zone.eq(zone) & saved.target.eq(target)].copy()
                part['timestamp'] = pd.to_datetime(part.timestamp, utc=True)
                cutoff = brussels_cutoff_timestamp(run_date)
                expected = set(pd.date_range(cutoff, periods=forecast_horizon_hours(run_date), freq='h'))
                for model, group in part.groupby('model'):
                    if set(group.timestamp) == expected and group.forecast_mw.notna().all():
                        complete.add(model)
            pending = model_keys - complete
            if not pending:
                print('Already complete', run_date, zone, flush=True)
                continue
            print('Recovering', run_date, zone, ','.join(sorted(pending)), flush=True)
            daily.zones = lambda zone=zone, config=config: {zone: config}
            output = daily.run_daily_forecast(run_date, model_keys=pending, clear_issues=False)
            if not output.empty:
                output['source'] = 'historical_mock_replay'
                store.append_forecasts(output, run_date.isoformat())
            for model in sorted(pending):
                count = 0 if output.empty else output[output.model.eq(model)].timestamp.nunique()
                report.append(dict(run_date=str(run_date), zone=zone, model=model, hours=count))
            # Preserve a result before proceeding to the next date.
            pd.DataFrame(report).to_csv(store.data_dir / 'backfill/offshore_recovery.csv', index=False)
    failed = [row for row in report if row['hours'] != forecast_horizon_hours(row['run_date'])]
    if failed:
        raise RuntimeError(f'{len(failed)} offshore model windows remain incomplete: {failed}')
    print('Recovered', len(report), 'model windows', flush=True)


if __name__ == '__main__':
    main()
