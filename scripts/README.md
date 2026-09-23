# Scripts map

| Folder | Role |
|--------|------|
| `model/` | OpenSees SteelMPF material + corotational truss |
| `postprocess/` | Cycle points, filter, resample, specimen plots |
| `calibrate/` | Individual + generalized L-BFGS, overlays, reports |
| `calibrate_single/` | One-specimen CLI (`calibrate_one_specimen.py` + `input.csv`) |
| `examples/` | Short, editable scripts (e.g. `simple_brb_steelmpf.py`) |

User-facing instructions: root [`README.md`](../README.md). Edit knobs in `config/calibration/` and `calibrate_single/input.csv`, not in these modules.
