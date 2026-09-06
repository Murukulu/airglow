# airglow

> You probably want to go [here](aifsv2/README.md) for medium range weather forecasting.

Two Rust crates from a summer of trying to get weather models running on [burn](https://burn.dev).

- [`aifsv2/`](aifsv2/): the actual project. ECMWF's AIFS single v2 as a Rust inference engine:
  open-data GRIB in, a 6 h forecast out as GRIB2. Go there, [that README](aifsv2/README.md) has
  everything.
- [`gnn_leffingwell_odor/`](gnn_leffingwell_odor/): the warm-up. A GCN on the Leffingwell odour
  dataset ([#8](https://github.com/Murukulu/airglow/issues/8)), done to learn burn's graph and
  tensor story before touching AIFS. It trains but *how well* was out of scope.

`nix develop` at the root gives the dev shell (eccodes, proj, libclang) both crates build in.

Apache-2.0.
