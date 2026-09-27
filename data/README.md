# Dataset setup - PeMSD4

Required file: `data/raw/pems04.npz`

| Property | Value (verified by `python -m src.data.validator`) |
|---|---|
| Format | NumPy `.npz` archive, key `data` |
| Shape | `[16992, 307, 3]` = time steps x sensors x channels |
| Channels | 0 = traffic flow (target), 1 = occupancy, 2 = speed |
| Interval | 5 minutes (16992 / 288 = 59 whole days, 1 Jan - 28 Feb 2018) |
| Region | San Francisco Bay Area (Caltrans PeMS District 4) |

Download: the file is distributed with the ASTGCN (Guo et al., AAAI 2019) code repository.

```bash
mkdir -p data/raw
curl -L -o data/raw/pems04.npz https://github.com/Davidham3/ASTGCN/raw/master/data/PEMS04/pems04.npz
```

`data/processed/` is intentionally empty. Preprocessing runs in memory on every pipeline run so the
steps are always reproducible from the raw file.
