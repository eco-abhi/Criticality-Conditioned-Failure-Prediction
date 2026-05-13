# part-class-variability

Research workspace for **criticality-aware inventory** modeling: a **hybrid** dataset that combines public supply-chain backbones (DataCo, UCI Online Retail) with **literature-informed synthetic augmentation** (BOM DAG, supplier panel, calibrated compliance outcomes). The canonical entry point is the CLI script `generate_synthetic_datasets.py`.

## Requirements

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/) for environments and runs

```bash
cd part-class-variability
uv sync
```

Optional dev tools (Jupyter, etc.):

```bash
uv sync --group dev
```

## Public data

1. **DataCo SMART Supply Chain** (`DataCoSupplyChainDataset.csv`)  
   Place the file at `data/DataCoSupplyChainDataset.csv`, or set **`DATACO_CSV_PATH`** to its location.  
   Sources: [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6.5) (doi:10.17632/8gx2fvg2k6.5); Kaggle dataset `shashwatwork/dataco-smart-supply-chain-for-big-data-analysis`.  
   Optional remote load (not recommended for reproducibility): set **`DATACO_FALLBACK_URL`** to a direct HTTPS URL to the CSV.

2. **UCI Online Retail**  
   Downloaded automatically at run time from the URL list in `generate_synthetic_datasets.py` (needs network on first run).

## Generate datasets

From the repo root:

```bash
uv run python generate_synthetic_datasets.py
```

Useful flags:

| Flag | Meaning |
|------|--------|
| `--output-dir PATH` | Default: `outputs/` under the repo |
| `--plots` | Write EDA figures under `<output-dir>/figures/` |
| `--n-parts N` | Override catalog size (default 5000) |
| `--demo` | In-memory DataCo-shaped stub **only** for smoke tests (no real DataCo file) |
| `--help` | Full CLI help |

Constants (ABC shares, BOM depth/fanout, compliance target rate, etc.) live at the top of `generate_synthetic_datasets.py`.

## Outputs

Written to **`outputs/`** (gitignored by default):

| Artifact | Description |
|----------|-------------|
| `part_catalog.csv` | Unified part catalog + synthetic features + BOM metrics |
| `supplier_history.csv` | Monthly supplier / OTD panel with rolling features |
| `compliance_outcomes.csv` | Calibrated compliance labels |
| `bom_graph.gpickle` | NetworkX BOM DAG (edges: component → assembly) |
| `data_dictionary.md` | Column provenance (public vs synthetic) |

## Other scripts

- **`bom_graph.py`** — Tiered BOM DAG generation and position features (loaded by the CLI via `importlib`).
- **`visualize_bom_graph.py`** — Histogram + tiered subgraph PNGs (`bom_graph_visualization_v2.png`, `bom_subgraph_tiered.png`) from `outputs/bom_graph.gpickle` and `outputs/part_catalog.csv`.
- **`data_tests.py`** — Quick sanity prints on exported CSVs and the BOM graph.
- **`_build_synthetic_notebook.py`** — Regenerates `synthetic_data_generator.ipynb` if you keep the notebook path in sync with the pipeline.

## License / data use

Respect the **DataCo** and **UCI** dataset terms when redistributing or publishing. This repository does not redistribute those CSVs; obtain them from the sources above.
