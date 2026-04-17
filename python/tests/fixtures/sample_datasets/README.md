# Functionality Testing Datasets

Use these `dataset_id` values with the API payload field `dataset_id`.

| Functionality to test | dataset_id | Source |
| --- | --- | --- |
| Cold-start model selection quality | `airline` | Built-in sktime dataset (`load_airline`) |
| Reasoning for monthly trend/seasonality | `m4_monthly_subset_like` | Generated local CSV fixture |
| Structural-break handling (COVID-like shock) | `m5_covid_period_like` | Generated local CSV fixture |
| Multi-seasonality behavior (hourly) | `etth1_like_hourly` | Generated local CSV fixture |
| Drift detection / memory refresh | `yahoo_s5_like_drift` | Generated local CSV fixture |
| Event-driven spikes + regime changes | `m5_promotional_events_like` | Generated local CSV fixture |

## Regenerate local fixtures

From `python/`:

```bash
python scripts/generate_functionality_datasets.py
```

Notes:

- `airline` does not need a CSV file.
- Other dataset IDs resolve to CSV files under this folder.
