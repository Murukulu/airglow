# GRIB → inference: the data pipeline for AIFS-single 2.0

> **Read `/Users/sai/Documents/projects/airglow/aifsv2/docs/grib-input-explained.md` first.**
> That document establishes _what the data is_ — the GRIB format, the contents of the three
> files in `data/`, the three grids and every shape transition, the
> prognostic/diagnostic/forcing provenance, and the anemoi call chain from file to model.
> This document assumes all of it and covers only _what to build in Rust and in what order_:
> the port decisions, the algorithms, and the build sequence with its oracles. Descriptive
> material that used to live here now lives there, and the cross-references below point at it
> rather than repeating it.

Everything upstream of the first `Linear`.
`/Users/sai/Documents/projects/airglow/aifsv2/src/` currently holds the model layers
(encoder, decoder, transformer, block, common, named_node_attributes) but nothing that
produces a tensor to feed them: `/Users/sai/Documents/projects/airglow/aifsv2/src/main.rs` loads safetensors into a one-field dummy
`Model`, `/Users/sai/Documents/projects/airglow/aifsv2/src/graph.rs` loads `GraphData` from the
extracted graph safetensors but nothing feeds it weather, and
`NamedNodeAttributesConfig::init` still carries `FIXME(saiputravu): Ingest graph data
correctly`.

This document specifies the missing layer — read GRIB, compute forcings, normalise,
assemble the 224-channel input, build the two bipartite graphs — and proposes a Rust
module design for it.

## Sources and conventions

Every path in this document is absolute, and every citation is
`/absolute/path/file.py:line`. The five roots that appear:

| root                                                                                                                  | what                                 |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `/Users/sai/Documents/projects/airglow/aifsv2`                                                                        | this repo                            |
| `/Users/sai/Documents/projects/anemoi/anemoi-core`                                                                    | graphs, models                       |
| `/Users/sai/Documents/projects/anemoi/anemoi-datasets`                                                                | dataset build                        |
| `/Users/sai/Documents/projects/anemoi/anemoi-inference`                                                               | runtime: rollout, state, schema      |
| `/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages` | earthkit, anemoi-models as installed |

Every number in §1–§7 was measured from `/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0.safetensors` and
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0_metadata.json` with `/Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_safetensors.py`, not read off
documentation. Reproduce any of them with the commands in the appendix.

### The extracted checkpoint payload

`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/` holds
two unrelated things. `.venv/` and `pyproject.toml` are a stale `uv init` scratch project —
ignore them except as a convenient environment. `anemoi-metadata/` is **checkpoint
payload**, extracted from the `.ckpt` zip, and is load-bearing:

| file                               | size        | what                                        |
| ---------------------------------- | ----------- | ------------------------------------------- |
| `anemoi-metadata/ai-models.json`   | 310,798 B   | the checkpoint metadata, unwrapped          |
| `anemoi-metadata/latitudes.numpy`  | 4,336,640 B | **raw little-endian f64, no `.npy` header** |
| `anemoi-metadata/longitudes.numpy` | 4,336,640 B | same                                        |

`4,336,640 = 542,080 × 8` exactly — the name is a misnomer, there is no numpy header to
skip. `np.fromfile(path, dtype='<f8')` and Rust's `std::fs::read` + a f64 cast both work
directly. §1 and §7 use these in preference to the checkpoint's float32 sin/cos encoding.

`ai-models.json` is content-identical (verified by equality, not by eye) to the single
nested value inside `.../data/aifs-single-mse-2.0_metadata.json`, which wraps it under the
key `'quiet_grub/anemoi-metadata/ai-models.json'`. Prefer the unwrapped file.

We are not porting anemoi's structure. Anemoi is modular because it serves every model in
the family; this serves exactly one checkpoint. The modularity that _is_ worth keeping is
isolated to two seams, called out in §9: where field data enters, and where the
checkpoint's schema is read.

---

## 1. The data contract

| quantity              | value                                     | measured from                                             |
| --------------------- | ----------------------------------------- | --------------------------------------------------------- |
| dataset variables     | 134, shape `[10228, 134, 1, 542080]`, 6 h | `metadata['dataset']`                                     |
| model input channels  | 106                                       | `data_indices.model.input.full`                           |
| model output channels | 120 = 92 prognostic + 28 diagnostic       | `data_indices.model.output`                               |
| forcings              | 14                                        | `data_indices.data.input.forcing`                         |
| multistep             | 2                                         | `config.training.multistep_input`                         |
| encoder input dim     | **224** = 2×106 + 12                      | `model.encoder.emb_nodes_src.weight [1024, 224]`          |
| node attribute dim    | **12** = 2×2 + 8                          | `model.encoder.emb_nodes_dst.weight [1024, 12]`           |
| data nodes            | **542,080** (N320 reduced Gaussian)       | `latlons_data [542080, 4]`                                |
| hidden nodes          | **40,320** (O96 octahedral)               | `latlons_hidden [40320, 4]`                               |
| encoder edges         | **748,348**                               | `model.encoder.trainable.trainable [748348, 8]`           |
| decoder edges         | **1,626,240** = 542,080 × 3               | `model.decoder.trainable.trainable [1626240, 8]`          |
| edge feature dim      | **11** = 1 + 2 + 8                        | `model.{encoder,decoder}.proc.lin_edge.weight [1024, 11]` |
| latent width          | 1024                                      | `config.model.num_channels`                               |

The node attribute arithmetic is
`attr_ndims = 2 * graph[nodes].x.shape[1] + num_trainable_params`
(`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/layers/graph.py:86`) = 2×2 + 8 = 12, and the
encoder input is `multi_step * num_input_channels + attr_ndims`
(`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/models/encoder_processor_decoder.py:229`)
= 2×106 + 12 = 224. §3.4 of `grib-input-explained.md` walks the full shape chain that
produces it.

### Node coordinate encoding

`latlons_data` is **not** raw lat/lon. `register_coordinates` stores
`torch.cat([sin(coords), cos(coords)], dim=-1)`
(`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/layers/graph.py:92`), so the four columns are:

```
[ sin_lat, sin_lon, cos_lat, cos_lon ]
```

Recovered by `atan2(x[:, :2], x[:, 2:])`
(`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/layers/graph.py:95-101`). Verified empirically against the checkpoint:

```
atan2(col0, col2) → [-89.784874, 89.784874] degrees   (latitude)
atan2(col1, col3) → [-179.99998, 179.85185] degrees   (longitude, wrapped to ±180)
col0² + col2² = 1.0
```

The latitude bound matches `latitudeOfFirstGridPointInDegrees = 89.785` reported by
`/Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_grib.py` for `/Users/sai/Documents/projects/airglow/aifsv2/data/lsm.grib` — same N320 grid, independently confirmed.
Note the longitude convention differs: this encoding wraps to ±180, GRIB reports
0…359.719.

**Consequence: the checkpoint already contains both grids**, so no Gaussian-latitude solver
is ever needed — see §3.2 of `grib-input-explained.md` for why that is not obvious from the
GRIB file alone. That fact drives the whole GRIB decision in §8.

### Prefer the f64 supporting arrays for the data grid

`latlons_data` is float32 and sin/cos-encoded, so recovering coordinates from it costs an
`atan2` and some precision. The same coordinates exist as raw f64 in
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/latitudes.numpy`
and `longitudes.numpy`. Measured against each other:

```
f64 latitudes            : 542,080 values, 640 distinct (= the 640 rows of `pl`)
                           range ±89.78487690721863
f64 longitudes           : 0 … 359.71875, row 0 = [0, 20, 40, …]  (360/18, and pl[0] = 18)
max |f64 − atan2(f32)|   : 4.08e-6°   (median 5.6e-7°) = 1.5e-5 of an N320 row spacing
```

Two reasons to use the f64 arrays:

1. They are in the **0…360 convention**, matching GRIB, so no wrap reconciliation when
   cross-checking against a decoded field.
2. Loading is `std::fs::read` plus a f64 cast — no zip, no safetensors, no numpy parsing.

**But not for building the graph.** An earlier draft argued the f64 arrays were also the
safer input to §7, on the grounds that a 4e-6° discrepancy is far too small to flip a
cutoff decision. That was wrong — measured, it flips ten of them:

| data-node coordinates      | radius        | E           | vs 748,348 |
| -------------------------- | ------------- | ----------- | ---------- |
| f64 arrays, f64 math       | 0.01170280194 | 748,358     | **+10**    |
| `atan2(latlons_data)`, f64 | 0.01170280194 | **748,348** | **exact**  |
| `atan2(latlons_data)`, f32 | 0.01170280858 | 748,358     | **+10**    |

At E ≈ 7.5e5 with ~19 sources per target, enough node pairs sit within 4e-6° of the cutoff
that the tie-breaks do not cancel. The reason the f32 path is the one that matches is
circular and obvious in hindsight: anemoi built the graph and wrote `latlons_data` from
the _same_ node coordinates, so `latlons_data` is definitionally what the builder saw.
Reproducing an exact integer means reproducing its inputs, not improving on them.

So: **`atan2(latlons_data)` for graph construction, f64 arrays for everything else** —
regrid targets, forcings, output geolocation. The two agree to 1.5e-5 of a row spacing,
which is irrelevant everywhere except a hard radius comparison. Note also that the
intermediate arithmetic must be f64 even when the inputs are f32; doing it in f32 moves
the radius in the 9th digit and costs the same ten edges.

The hidden (O96) grid has no supporting array, so `latlons_hidden` remains the only
source for those 40,320 nodes; `atan2` there is unavoidable.

### Provenance — three sources, and they do not overlap

Every tensor the model consumes arrives one of three ways: **read** from the checkpoint,
**computed** at startup from coordinates, or **retrieved** as field data. Nothing is
sourced two ways, and the three are easy to conflate because they end up concatenated into
the same rows.

| tensor                              | cols | source                 | notes                                                 |
| ----------------------------------- | ---- | ---------------------- | ----------------------------------------------------- |
| node attrs, `latlons_{data,hidden}` | 0–3  | **read**               | f32 `sin/cos` of the grid. Static forever.            |
| node attrs, `trainable`             | 4–11 | **read**               | `[N, 8]` learned weights. Irreducible.                |
| `edge_index`                        | —    | **read _or_ computed** | in the `.ckpt` graph, absent from the safetensors; §7 |
| edge attrs, `edge_length/_dirs`     | 0–2  | **read _or_ computed** | ditto, and stored already unit-std normalised         |
| edge attrs, `trainable`             | 3–10 | **read**               | `[E, 8]`, and its row count _defines_ E               |
| the 106 input channels              | —    | **retrieved**          | 97 from GRIB, 9 computed from date+grid (§3)          |

Three consequences worth stating outright, because each one is a question that keeps
getting re-asked:

**Node attributes have nothing to do with GRIB.** All 12 columns are a checkpoint read.
Four say _where_ a node is; eight are a learned per-node embedding — `[542080, 8]` is 4.3M
fitted parameters encoding stable local character, not weather. A forecast changes 184 of
the 224 encoder input channels (92 prognostics × 2 timesteps); these 12 are not among
them. You could run the model on a grid with no weather on it and these columns would be
identical.

**Edges are never retrieved, and only conditionally computed.** Which artefact you start
from decides this. The `.safetensors` extract holds the 8 trainable columns per edge but
no connectivity, so working from it means rebuilding `edge_index` geometrically at every
startup (§7). The `.ckpt` — what anemoi itself loads — carries the built graph, so working
from it means extracting five tensors and skipping §7 entirely. The trainable tensors'
row counts, 748,348 and 1,626,240, are what makes the rebuild path verifiable rather than
a leap of faith: they pin the answer exactly without containing it.

**The `.numpy` supporting arrays are not the node-attribute source.** They are f64
coordinates used for _geometry_ — the kd-tree in §7 and the regrid target in §8. The
node-attribute columns are the f32 `sin/cos` encoding in the safetensors, which differs by
up to 4e-6° (measured above). Both describe the same 542,080 points; use the f64 arrays
wherever precision compounds and the checkpoint tensor wherever it is fed to the model
verbatim. The safetensors is mandatory regardless — the 8 trainable columns exist nowhere
else.

---

## 2. Where the 134 variables come from

**Moved.** The role partition (92 prognostic / 28 diagnostic / 14 forcing), the provenance
flags, the full variable lists, and the measured join showing which 97 of the 106 input
channels are retrievable from the sample GRIB files are all in §4 of
`/Users/sai/Documents/projects/airglow/aifsv2/docs/grib-input-explained.md`.

Three things that are decisions rather than facts, and so stay here:

**Every retrievable variable carries its own MARS request**, so "how do I fetch this" is a
lookup rather than a guess — 125 of the 134 have a `mars` block in
`dataset.variables_metadata`:

```json
"2t":  {"mars": {"class": "od", "stream": "oper", "type": "an", "levtype": "sfc",
                 "param": "2t", "step": 0, "time": 0, "date": 20160101}}
"lsm": {"constant_in_time": true, "mars": {..., "param": "lsm", "step": 0}}
"tp":  {"mars": {..., "type": "fc", "param": "tp", "step": 6, "time": 1800},
        "period": ["0:00:00", "6h"], "process": "accumulation"}
```

Note `tp` is `type: fc` at `step: 6` with an accumulation period, not an analysis — the six
`process: "accumulation"` variables (`cp`, `ro`, `sf`, `ssrd`, `strd`, `tp`) need different
retrieval from the instantaneous ones. All six are diagnostic, so this only matters for
output (§11).

**Treat the checkpoint flag as authoritative over `inference.yaml`.** `wmb` is
`constant_in_time: true` in the checkpoint but absent from the
`constant_fields: [z, sdor, slor, lsm]` patch at
`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:23-25`. The checkpoint
wins; `wmb` is fetched once and reused.

**The constant/dynamic split within the computed forcings drives the rollout.**
`cos/sin_latitude` and `cos/sin_longitude` are both `computed_forcing` and
`constant_in_time`, so they are computed once for the grid; `cos/sin_julian_day`,
`cos/sin_local_time` and `insolation` are recomputed every step. §11 depends on that
distinction. `anemoi-inference` exposes the set as `model_computed_variables`
(`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/metadata.py:330`):
"the initial conditions variables that need to be computed and not retrieved".

---

## 3. Computed forcings

Nine of the fourteen forcings are pure functions — no network, no files. All formulas from
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/sources/forcings.py`, class `ForcingMaker`.

These citations are into the _dataset-build_ stack, but they govern inference too:
`ComputedForcings.load_forcings_array`
(`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/forcings.py:139-160`)
reaches the same earthkit source, calling
`ekd.from_source("forcings", source, date=dates, param=self.variables)` over an
`UnstructuredGridFieldList.from_values(latitudes, longitudes)` taken from the current
state. One implementation, both paths — so a Rust port that matches these formulas matches
training and inference simultaneously. The returned array is ordered `(variable, date,
values)` (`forcings.py:177`).

### Position-only (constant across time, computed once)

```
cos_latitude   = cos(lat_rad)        :99      sin_latitude   = sin(lat_rad)     :106
cos_longitude  = cos(lon_rad)        :120     sin_longitude  = sin(lon_rad)     :127
```

### Date-only (constant across the grid)

```
julian_day     = (date − Jan 1 of date.year) as days + seconds/86400            :142
cos_julian_day = cos(julian_day / 365.25 · 2π)                                  :152
sin_julian_day = sin(julian_day / 365.25 · 2π)                                  :156
```

`julian_day` is a **day-of-year offset, zero-based** — Jan 1 00:00 gives 0.0, not the
astronomical Julian Day Number. The divisor is 365.25 regardless of leap year.

### Date × position

```
hours_since_midnight = (date − midnight_of_that_day) in hours
local_time     = (lon/360 · 24 + hours_since_midnight) mod 24                   :160
cos_local_time = cos(local_time / 24 · 2π)                                      :171
sin_local_time = sin(local_time / 24 · 2π)                                      :175
```

**Unit trap:** `local_time` consumes longitude in **degrees** (`lon/360`), while
`cos/sin_longitude` consume **radians**, and the node attributes in §1 are radians again.
Three conventions in one pipeline. Pick one internal representation (radians) and convert
at the call site.

### Insolation

`insolation` is an alias for `cos_solar_zenith_angle` (`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/sources/forcings.py:179-180`), implemented
in `/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/utils/meteo.py`. Full algorithm, worth transcribing exactly because
it is the only forcing with real substance:

```python
# /Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/utils/meteo.py:32  solar_declination_angle(date) -> (declination°, time_correction h·°)
angle = julian_day(date) / 365.25 * 2π                       # /Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/utils/meteo.py:20,23

declination = ( 0.396372
              - 22.91327·cos(angle)  + 4.025430·sin(angle)
              -  0.387205·cos(2·angle) + 0.051967·sin(2·angle)
              -  0.154527·cos(3·angle) + 0.084798·sin(3·angle) )

time_correction = ( 0.004297
                  + 0.107029·cos(angle) - 1.837877·sin(angle)
                  - 0.837378·cos(2·angle) - 2.340475·sin(2·angle) )

# /Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/utils/meteo.py:56  cos_solar_zenith_angle(date, lat°, lon°)
solar_angle  = deg2rad((date.hour − 12)·15 + lon° + time_correction)
zenith       = sin(declination_rad)·sin(lat_rad)
             + cos(declination_rad)·cos(lat_rad)·cos(solar_angle)
return clip(zenith, 0.0, None)                               # /Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/data/utils/meteo.py:101
```

Three details that are easy to lose: the truncated Fourier series coefficients are
literal; `solar_angle` uses `date.hour` only (**minutes are discarded**); and the result is
clipped at zero, so night is exactly 0, not negative.

Note `insolation` is in the normaliser's `none` list (§4) — it is fed to the model raw.

---

## 4. Pre-processors

> **Two different things share this name.** This section is about the **anemoi-models**
> pre-processors: tensor-level transforms (imputer, normaliser) whose coefficients are
> baked into the checkpoint and configured under `config.data.processors`. There is also
> an **anemoi-inference** `pre_processors/` package
> (`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/pre_processors/`:
> `mask.py`, `forward_transform_filter.py`, `extract.py`, `no_missing_values.py`) which
> operates on a `State` — a dict of _named fields_ — and is configured in
> `/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml`. Different layer,
> different data structure, different config file. `apply-mask` and
> `cos_sin_mean_wave_direction` in that YAML are the second kind; everything below is the
> first.

Not yet implemented in `/Users/sai/Documents/projects/airglow/aifsv2/src/`. Execution
order is the dict order of
`config.data.processors`: `conditional_nan_postprocessor`, `const_imputer`, `normalizer`.

### ConstantImputer

Fill value `0` for 17 variables: `cos_mwd sin_mwd mwp swh h1012 h1214 h1417 h1721 h2125
h2530 cdww wmb swvl1 swvl2 ro sd snowc`. Config-driven; no tensors in the checkpoint.

Mostly wave and soil parameters — undefined over land or over ice, hence NaN in the source
data.

### InputNormalizer — a pure affine map

This is the single biggest simplification available. The normalisation _method_
(`mean-std`, `std`, `max`, `none`) is resolved at training time into two coefficient
vectors (`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/anemoi/models/preprocessing/normalizer.py:71-95`):

```
mean-std :  _norm_mul = 1/σ        _norm_add = −μ/σ
std      :  _norm_mul = 1/σ        _norm_add = 0
max      :  _norm_mul = 1/max      _norm_add = 0
min-max  :  _norm_mul = 1/(max−min)  _norm_add = −min/(max−min)
none     :  _norm_mul = 1          _norm_add = 0
```

so `transform` is just (`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/anemoi/models/preprocessing/normalizer.py:162-166`):

```python
x.mul_(_norm_mul[idx]).add_(_norm_add[idx])
```

**Rust needs no method dispatch whatsoever.** Load four tensors:

```
pre_processors.processors.normalizer._norm_mul   [134] F32
pre_processors.processors.normalizer._norm_add   [134] F32
pre_processors.processors.normalizer._input_idx  [106] I32
pre_processors.processors.normalizer._output_idx [120] I32
```

The index selection at `/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/anemoi/models/preprocessing/normalizer.py:163-166` keys off the tensor's last dimension: 106
channels → use `_input_idx`, 134 → use the full vectors, explicit `data_index` → use that.
Replicate that branch or, better, resolve it once at load time since our shapes are fixed.

`post_processors.processors.normalizer.*` are the same four tensors for the inverse
(`x = (x − _norm_add)/_norm_mul`). There is also a `*_tendencies` pair, unused on this
config.

### ConditionalNaNPostprocessor

`nan: [snowc, ro]`, `remap: sd` — re-introduces NaN in `snowc` and `ro` wherever the
remap variable indicates, undoing the imputation for output fidelity.

---

## 5. `_assemble_input`

`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/models/encoder_processor_decoder.py:186-207`, with
the sharding branches removed (single device — `grid_shard_shapes=None`,
`model_comm_group=None` collapses `_get_shard_shapes` and every `shard_tensor` to
identity).

Entry shape after `predict_step` inserts the dummy ensemble axis (`:448`):

```
x : [batch, time=2, ensemble=1, grid=542080, vars=106]
```

### Where that tensor comes from

The model side above starts mid-stream. §5.2 of
`/Users/sai/Documents/projects/airglow/aifsv2/docs/grib-input-explained.md` traces the caller
(`tensors.py:143-172`, `runner.py:443`) and §3.4 diagrams the shapes. Two properties of that
construction are port decisions rather than description:

**The working layout is `(time, vars, grid)`** and only becomes `(time, grid, vars)` at the
last moment. Assembling directly in the final layout is fine, but if you cross-check against
a Python dump, know which side of that swap you are on.

**The NaN fill is a guard, and must be ported.** `tensors.py:168-172` asserts every one of
the 106 channels was written, naming the missing variables if not. Cheap, and it catches a
forgotten forcing immediately — see §11 for the full `check[]` invariant this is part of.

### Skip tensor

```python
x_skip = x[:, -1, ...]  # :187   last timestep only
x_skip = _apply_truncation(x_skip)  # :189   NO-OP on this checkpoint
```

`_apply_truncation` (`:164`) only acts if `A_down`/`A_up` are set, which requires
`config.hardware.files.truncation` / `truncation_inv`. Both are `null` here — verified —
so the whole truncation path is dead code for AIFS-single 2.0. Skip it entirely; add it
back only if a future checkpoint sets those files.

### Latent tensor

```python
x_data_latent = cat(
    [
        rearrange(
            x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"
        ),
        node_attributes_data,  # [542080, 12]
    ],
    dim=-1,
)  # :198-204
```

Result `[542080, 224]` at batch=ensemble=1.

**`(time vars)` means time is the outer index.** Channel `i·106 + j` is variable `j` at
timestep `i`; channels 0…105 are t−6h, 106…211 are t. Reversing this produces a tensor of
the right shape that silently yields garbage — there is no shape check that catches it.

Node attributes occupy the **last** 12 channels, laid out
`[sin_lat, sin_lon, cos_lat, cos_lon, trainable_0…trainable_7]` — from
`NamedNodesAttributes.forward` (`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/layers/graph.py:107-112`) delegating to
`TrainableTensor.forward` (`:37-45`), which does
`cat([repeat(latlons), repeat(trainable)])`.

`TrainableTensor.forward` also handles batching by `einops.repeat(x, "e f -> (repeat e) f",
repeat=batch_size)` — tiling along the node axis, matching `expand_edges` in
`/Users/sai/Documents/projects/airglow/aifsv2/src/graph.rs`. At batch=1 it is a no-op.

### Hidden side

```python
x_hidden_latent = node_attributes(hidden, batch_size)  # :362  → [40320, 12]
```

Pure lookup, no data. That is the entire encoder destination input.

### `x_data_latent` survives the encoder unchanged

Line `:366` reads `x_data_latent, x_latent = self._run_mapper(self.encoder, ...)`, which
looks like the encoder overwrites its source with a 1024-wide embedding. It does not.
`GraphTransformerForwardMapper.forward`
(`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/layers/mapper.py:599-621`)
ends with:

```python
return x[0], x_dst  # :621  x[0] is the *input* tensor, not the embedding
```

`ForwardMapperPreProcessMixin.pre_process` (`:135`) does apply `emb_nodes_src` internally,
but that embedding is consumed by the attention and then discarded — the source slot
returned to the caller is the original argument. So `x_data_latent` is the same
`[542080, 224]` tensor before and after the encoder, and the decoder at `:391` receives it
as its _destination_ input and re-embeds it from scratch with its own
`emb_nodes_dst.weight [1024, 224]`.

The checkpoint confirms this without reading any code — the shape would be `[1024, 1024]`
if the decoder consumed the encoder's latent:

```
model.encoder.emb_nodes_src.weight        [1024, 224]   data nodes, 2x106 + 12
model.encoder.emb_nodes_dst.weight        [1024, 12]    hidden nodes
model.decoder.emb_nodes_dst.weight        [1024, 224]   <- data nodes again, raw
model.decoder.node_data_extractor.1.weight [120, 1024]   -> the 120 output channels
```

Two consequences for the port. `x_data_latent` must stay live across the processor, so
budget `542080 × 224 × 4 B ≈ 486 MB` at f32 for the whole forward pass — it is not
free-able after the encoder. And it is a **pass-through, not a skip connection**: no
learned state flows along it, so there is no ordering hazard, only a lifetime one. The
genuine skip connections are `x_latent_proc + x_latent` (`:386`) and the prognostic
residual in §6.

Full width chain, batch = ensemble = 1:

```
x_data_latent   [542080,  224] ─────────────────────────────┐  (unchanged)
x_hidden_latent [ 40320,   12] ──encoder──> x_latent [40320, 1024]
                                              │
                                          processor + skip
                                              ▼
                                        x_latent_proc [40320, 1024]
                                              │
                    decoder(src=x_latent_proc, dst=x_data_latent) <┘
                                              ▼
                                        x_out [542080, 120]
```

---

## 6. `_assemble_output`

`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/models/encoder_processor_decoder.py:209-227`.

```python
x_out = rearrange(
    x_out, "(batch ensemble grid) vars -> batch ensemble grid vars"
)  # :211
x_out[..., _internal_output_idx] += x_skip[..., _internal_input_idx]  # :222
for bounding in boundings:
    x_out = bounding(x_out)  # :224
```

The residual is a **scatter-add across two different index spaces**:
`_internal_output_idx = data_indices.model.output.prognostic` (92 indices into the
120-channel output) and `_internal_input_idx = data_indices.model.input.prognostic` (92
indices into the 106-channel input). They are not equal — measured,
`model.input.prognostic` starts `[0,1,2,3,4]` while `model.output.prognostic` starts
`[2,3,4,5,6]`. Both arrays must be read from metadata, not assumed.

Only prognostics get the residual; the 28 diagnostics are predicted outright.

Also in the forward pass, `latent_skip: True` gives
`x_latent_proc = x_latent_proc + x_latent` (`:386`), and `grid_skip: 0`.

### Boundings

From `config.model.bounding`, applied **in list order** — four of them, and the order
matters because `FractionBounding` reads a variable the earlier entries have already
clamped:

| # | bounding                 | n  | variables                                                                                      |
| - | ------------------------ | -- | ---------------------------------------------------------------------------------------------- |
| 1 | `ReluBounding`           | 26 | `tp ro tcw ssrd sd`, all 12 prognostic `q_*`, `swh mwp`, `h1012 h1214 h1417 h1721 h2125 h2530` |
| 2 | `HardtanhBounding(0, 1)` | 4  | `tcc swvl1 swvl2 snowc`                                                                        |
| 3 | `FractionBounding`       | 2  | `cp sf` — as a fraction of `tp`                                                                |
| 4 | `FractionBounding`       | 3  | `lcc mcc hcc` — as a fraction of `tcc`                                                         |

`FractionBounding` is the one with a dependency — each entry names its referent in
`total_var` (`"tp"` and `"tcc"` respectively) and constrains its variables to
`[min_val, max_val] × total_var`. So `cp`/`sf` are bounded by the already-Relu'd `tp`, and
the three cloud covers by the already-Hardtanh'd `tcc`. Applying these out of order gives
wrong answers silently. Read `total_var` from the config rather than inferring the
referent from the variable names.

---

## 7. Graph construction

### The graph is already in the `.ckpt` — none of this is mandatory

An earlier draft opened "neither `edge_index` is in the checkpoint … both must be computed
geometrically at startup." That is true of
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0.safetensors`, which
holds only the tensors of the `state_dict`, and false of
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0.ckpt`, which is what
anemoi actually loads. `AnemoiModelInterface.graph_data` is a pickled `HeteroData` carrying
the whole thing:

```
HeteroData(
  data  ={ x=[542080, 2], node_type='AnemoiDatasetNodes',       area_weight=[542080, 1] },
  hidden={ x=[40320, 2],  node_type='ReducedGaussianGridNodes' },
  (data,   to, hidden)={ edge_index=[2, 748348],   edge_type='CutOffEdges',
                         edge_length=[748348, 1],  edge_dirs=[748348, 2] },
  (hidden, to, data  )={ edge_index=[2, 1626240],  edge_type='KNNEdges',
                         edge_length=[1626240, 1], edge_dirs=[1626240, 2] },
  (hidden, to, hidden)={},
)
```

`edge_index` is already `[src, dst]` — the `flip` is baked in. `edge_length` and `edge_dirs`
are already **unit-std normalised**: both have `std(unbiased) == 1.000000` exactly, which
independently confirms the §7.7 reading that `torch.std` is taken over the _flattened_
tensor (one scalar for both `edge_dirs` columns) with the n−1 estimator. `data.x` and
`hidden.x` are the raw `[N, 2]` lat/lon in radians, f32 — the pre-`sin/cos` coordinates the
builder saw, and the thing `register_coordinates` later encodes into `latlons_data`.

`x` also carries `area_weight` on the data nodes (`SphericalAreaWeights`, `unit-max`) — a
training-loss weight, unused at inference.

**So the practical choice is: extract, or rebuild.** Extracting is a one-time script that
dumps five tensors to a file the Rust side memory-maps; it eliminates the kd-tree, the
haversine, the Rodrigues rotation and the normalisation entirely. Rebuilding keeps the
runtime dependency-free and lets a future checkpoint on a different grid work without a
Python round-trip. Rebuilding is _verifiable_ (below), so it is a real option rather than a
leap — but it is now a choice, not a requirement, and phase 2 of §10 should be read in that
light.

Note the extraction is not free: `torch.load(..., weights_only=False)` unpickles the entire
994 MB model object and therefore needs `torch`, `torch_geometric`, `anemoi.models` and
stubs for whatever the pickle references but the environment lacks (`flash_attn` here).
That cost is paid once, offline.

### Rebuilding reproduces the graph exactly

Measured against the extracted `edge_index`, as a set of `(src, dst)` pairs:

| data/hidden coordinates fed to the builder     | E       | missing | extra |               |
| ---------------------------------------------- | ------- | ------- | ----- | ------------- |
| `atan2(latlons_{data,hidden})`, f64 math       | 748,348 | 0       | 0     | **identical** |
| `graph.x` / `hidden.x` (the stored f32 coords) | 748,350 | 0       | 2     |               |
| f64 `latitudes.numpy` / `longitudes.numpy`     | 748,358 | —       | —     |               |

The middle row is the surprise: feeding the builder the _exact_ coordinates it was built
from still lands two edges long, because `torch_cluster.radius` does its comparison
differently from `scipy.cKDTree.query_ball_point`. The `atan2` path matching to the edge is
therefore partly luck, and the right way to hold it is as a **regression check on a fixed
checkpoint**, not as a guarantee that any radius search will agree. If a rebuild lands
within a handful of edges, the algorithm is right and the difference is boundary
tie-breaking.

### The verification oracle

The per-edge trainable tensors pin the expected edge counts exactly:

| subgraph                | builder                              | expected E    |
| ----------------------- | ------------------------------------ | ------------- |
| data → hidden (encoder) | `CutOffEdges(cutoff_factor=0.6)`     | **748,348**   |
| hidden → data (decoder) | `KNNEdges(num_nearest_neighbours=3)` | **1,626,240** |

`1,626,240 = 542,080 × 3` confirms k-NN with one edge per (target, neighbour) pair. The
encoder count is not derivable a priori — it depends on the cutoff radius — which makes it
a genuinely strong test: if the Rust builder emits 748,348 edges, the radius, the
reference distance and the metric are all correct simultaneously.

### Algorithm

Feed the builder `atan2(latlons_data)` for the data nodes and `atan2(latlons_hidden)` for
the hidden nodes, in f64 arithmetic — **not** the f64 supporting arrays, which miss the
exact edge count by 10; see §1. Everything below is in radians
on the unit sphere; the 0…360 vs ±180 longitude convention does not matter to any of it,
since every operation goes through `cos(lon)`/`sin(lon)`.

All citations into `/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/`.

**1. To Cartesian** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/generate/transforms.py:50-53`), unit sphere, lat/lon in radians:

```
x = cos(lat)·cos(lon)      y = cos(lat)·sin(lon)      z = sin(lat)
```

**2. Grid reference distance** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/utils.py:65-85`):

```
xyz   = to_cartesian(coords)
dists = kneighbors(xyz, n_neighbors=2)        # self + nearest
ref   = max(dists[dists > 0])
```

The **maximum** over all nodes of each node's nearest-neighbour distance — a worst-case
spacing, dominated by the sparsest part of the grid (near the poles for a reduced Gaussian
grid). Euclidean chord distance in 3-D, not great-circle.

Computed once per node set at registration (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/nodes/builders/base.py:64`) and cached as
`_grid_reference_distance`.

**3. Encoder edges** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/builders/cutoff.py:104-132`):

```
radius     = 0.6 × reference_distance(TARGET nodes = hidden)      :104-124
edge_index = radius_search(source, target, r=radius,
                           max_num_neighbors=64)                  :131
edge_index = flip(edge_index, [0])                                :132
```

Two things to get right: the radius comes from the **target** (hidden/O96) node set, not
the source; and the `flip` converts PyG's `[target_idx, source_idx]` convention to
`[src, dst]`, which is the `[2, E]` row order `GraphData` in `/Users/sai/Documents/projects/airglow/aifsv2/src/graph.rs` already expects. The
`max_num_neighbours=64` cap truncates any target with more than 64 in-radius sources.

**4. Decoder edges** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/builders/knn.py:71-73`):

```
edge_index = knn(source, target, k=3)   then flip([0])
```

k-NN over the same Cartesian coordinates, euclidean.

**5. `edge_length`** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/attributes.py:86-92` → `/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/utils.py:106-128`) — **haversine on
lat/lon in radians**, deliberately a different metric from the euclidean one used to
_build_ the edges:

```
a = sin²(Δlat/2) + cos(lat_i)·cos(lat_j)·sin²(Δlon/2)
d = 2·atan2(√a, √(1−a))
```

**6. `edge_dirs`** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/attributes.py:94-99` → `/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/directional.py:65-98`) — rotate
each source point by the rotation that carries its target to the north pole, then take the
horizontal direction:

```
north       = [0, 0, 1]                                   /Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/directional.py:15
v_unit      = direction_vec(target_xyz, north)            :85   (normalised cross product)
θ           = acos(clamp(target_xyz · north, −1, 1))      :86-88
src_rotated = rotate_vectors(source_xyz, v_unit, θ)       :91   (Rodrigues)
direction   = direction_vec(src_rotated, north)           :94
return direction[:, :2]                                   :98   (third component ≡ 0)
```

Rodrigues (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/directional.py:32-63`):
`v·cosθ + (axis × v)·sinθ + axis·(v·axis)·(1−cosθ)`.
`direction_vec` (`:18-30`) is `normalize(cross(points, reference))` with an ε nudge when
the cross product degenerates (points colinear with the pole).

**7. `unit-std` normalisation** (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/normalise.py:37-41, 91-124`). `EdgeLength` and
`EdgeDirection` both inherit `norm_by_group = False` (`/Users/sai/Documents/projects/anemoi/anemoi-core/graphs/src/anemoi/graphs/edges/attributes.py:35`), so:

```
values / torch.std(values)
```

`torch.std` over the **whole tensor flattened** — for the 2-column `edge_dirs` that is one
scalar shared by both columns, not per-column. And `torch.std` defaults to the _unbiased_
(n−1) estimator; the difference from the biased one is ~1e-6 relative at E≈10⁶, but state
the choice rather than discover it.

Final per-subgraph edge features: `[edge_length(1), edge_dirs(2), trainable(8)]` → 11,
matching `lin_edge.weight [1024, 11]`. Concatenation order follows
`config.model.{encoder,decoder}.sub_graph_edge_attributes = ['edge_length', 'edge_dirs']`.

### Implementation note

A spatial index over 542,080 + 40,320 points in 3-D is required — a kd-tree is the
straightforward choice; a fixed-radius query for the encoder and a k=3 query for the
decoder. Both are one-time startup costs, cacheable to disk keyed on the checkpoint hash.

---

## 8. Reading GRIB from Rust

### The crate landscape does not cover this grid

| crate                                                            | editions       | grids                                                              |
| ---------------------------------------------------------------- | -------------- | ------------------------------------------------------------------ |
| [`grib`](https://github.com/noritada/grib-rs) (grib-rs)          | GRIB2 only     | 3.0, 3.1, 3.20, 3.30, 3.40 Gaussian — all **"regular grids only"** |
| [`gribberish`](https://crates.io/crates/gribberish)              | GRIB2          | reduced-Gaussian support unconfirmed                               |
| [`grib-reader`](https://docs.rs/grib-reader/latest/grib_reader/) | GRIB2          | regular lat/lon, Lambert, polar stereographic                      |
| [`eccodes`](https://github.com/ScaleWeather/eccodes)             | via libeccodes | everything                                                         |

`/Users/sai/Documents/projects/airglow/aifsv2/data/lsm.grib` is **GRIB edition 1** on `reduced_gg`, so grib-rs excludes it twice over.
Even for GRIB2 inputs, "Gaussian, regular grids only" means the N320 reduced grid the
model runs on is unsupported. The answer to "Rust has GRIB crates, why would I do FFI" is
that **none of them decode the grid this model uses** — not that FFI is inherently better.

### Packing is easy for `lsm.grib` and hard for everything else

The measured packing inventory is in §1.3 and §2 of
`/Users/sai/Documents/projects/airglow/aifsv2/docs/grib-input-explained.md`. The decision it
forces, in one line: **`lsm.grib` is the only `grid_simple` message in the repository, and
all 137 messages that actually feed the model are `grid_ccsds`.**

CCSDS is Adaptive Entropy Coding (libaec/szip), not a scale-and-shift. Hand-rolling it is
not a few hundred lines, and that moves the FFI trade below decisively.

Two packing edge cases have to be designed around either way:

- **`bitsPerValue = 0`** on the six accumulation variables. A reader that assumes ≥1 bit
  divides by zero or reads nothing. All six are diagnostic-only, so they can also just be
  dropped on input.
- **Bitmaps** on `vsw` and every wave field. These are precisely the fields the
  `ConstantImputer` (§4) fills, so masked points must reach the imputer as **NaN**. Note
  that this is an active step, not a default: eccodes hands back the full expanded array
  with its `missingValue` sentinel — **9999.0** unless you change it — at the masked points,
  and 9999 passes a NaN-only imputer untouched. earthkit sets the sentinel and rewrites it to
  NaN; a Rust reader must do the same. §1.4 of `grib-input-explained.md` has the measured
  numbers and the earthkit source.

### The open-data grid is not the model grid

`regular_ll` 1440 × 721 against N320's 542,080. **A regridding step is mandatory**, and §9
has no home for it. §3.3 of `grib-input-explained.md` establishes that this is a cached
sparse matrix multiply rather than an interpolation algorithm, which shrinks the task but
does not remove it from the critical path.

The consequence for retrieval ordering is that `lsm.grib` being on a different grid from the
forecast files is _not_ an inconsistency. Open data is a derived product: ECMWF interpolates
native N320 output onto 0.25° and requantises before publishing. `lsm.grib` came from MARS
upstream of that, and it is consumed _after_ the regrid — the `opendata` input plugin
retrieves and regrids, and only then do `pre_processors` run, so `MaskValues` sees
542,080-point fields and needs a 542,080-point mask.

**Prefer `lsm.grib` for the `lsm` forcing channel too**, not just for the mask: native
resolution, no interpolation of a land fraction across coastlines, and the measured
quantisation cost of the open-data copy is severe (§2.3 there). `sdor`, `slor`, `z` and
`wmb` have no native copy and must go through the regrid.

Note also that the GRIB reader only has to produce **values in stored order** — never
coordinates — because the checkpoint ships the N320 grid as raw f64 (§1 here, §3.2 there).
No Gaussian quadrature is required anywhere in this pipeline.

### The reference implementation agrees

This is not an inference from first principles — it is what anemoi actually does with this
exact file. `MaskValues.__init__`
(`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/pre_processors/mask.py:50-57`)
resolves its mask in three branches:

```python
if mask in self.checkpoint.supporting_arrays:  # :50
    mask_array = self.checkpoint.supporting_arrays[mask]
elif mask.endswith(".npy"):  # :52
    mask_array = np.load(mask)
elif mask.endswith(".grib") or mask.endswith(".nc"):  # :54
    mask_array = ekd.from_source("file", mask)[0].to_numpy(flatten=True)
```

This checkpoint's `supporting_arrays` holds only `latitudes`/`longitudes`, so
`path: lsm.grib` in
`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml` takes the third branch:
one message, flattened, values only. **No coordinates are read from the GRIB at all** —
exactly the scope proposed below.

One behavioural note that matters if you compare against Python: earthkit/eccodes returns
the 542,080 _stored_ values for a reduced Gaussian field, whereas pygrib expands to
640×1280 = 819,200 by default. `/Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_grib.py`
already suppresses that expansion (`expand_grid(False)`), so its statistics are directly
comparable to what the reference implementation sees. Rust should produce the stored 542,080.

### What a real pull actually covers

The join of the 97 retrievable input channels against every message in the three files is
in §4.5 of `grib-input-explained.md`: 90 match directly, 6 need aliasing, `sd` is absent.
Three things to design for follow from it.

**Pick a routing vocabulary and keep a rename table beside it.** Six of the 97 are not
addressable by the model's variable name. anemoi resolves this by naming fields from the
ecCodes `mars` namespace — `param` or `f"{param}_{levelist}"` — which produces `vsw_1`,
`sot_1` and `mwd` where the model wants `swvl1`, `stl1` and `cos_mwd`/`sin_mwd`; §5.4 of
`grib-input-explained.md` traces the four translation layers. Two consequences for the §9
`FieldMessage`: its `LevelType` must commit to either the MARS vocabulary (`pl`, `sfc`,
`sol`, `o2d`) or the ecCodes one (`isobaricInhPa`, `surface`, `soilLayer`,
`heightAboveGround`, `heightAboveSea`) and convert at the boundary — they are not the same
set — and the rename table is unavoidable either way, because `vsw_1 → swvl1` is not derivable
from any key.

Note that `use_grib_paramid: true`
(`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:27`) does **not** govern
this. It is consulted only when building outgoing MARS requests (`metadata.py:868`) and when
patching output archive requests (`outputs/gribfile.py:306`).

**`sd` (snow depth) is in neither file** — a prognostic input, the `remap` key for
`ConditionalNaNPostprocessor` (§4) and one of the three `apply-mask` targets, so not
droppable. Either the download's parameter list was short or open data does not serve it at
0.25°; settle this against the catalogue before building against these files.

**45 messages are surplus and only one timestep is present.** The surplus is the 28
diagnostics plus 14 `gh` fields redundant with `z` on pressure levels; everything is
`step=0`, so `multistep=2` needs a second pull at −6h.

### Recommendation

**Take the `eccodes` crate.** The earlier reasoning — one simple encoding, so hand-roll it
— was measured against `lsm.grib` alone and does not survive contact with the actual
inputs. A hand-written reader now needs GRIB1 _and_ GRIB2 section walking, `grid_simple`
_and_ `grid_ccsds`, bitmaps, and the `bitsPerValue = 0` degenerate case. CCSDS alone means
either a libaec dependency or an AEC implementation, at which point the "no C build
dependency" argument has already been conceded and bought nothing.

What survives from the original scope is the more valuable half: **no grid-geometry
decoding**. Read values in stored order and take coordinates from the checkpoint. Verified
— `lsm.grib`'s eccodes-synthesised coordinates match `latitudes.numpy` / `longitudes.numpy`
elementwise to 2.8e-14, in the same order, and `pl` is recoverable from the checkpoint
latitudes alone. So value `i` in stored order is node `i`; no lookup table, no scan-mode
logic. The Legendre-root Newton iteration is never needed.

Scope narrows in one direction, though: **GRIB1 and `grid_simple` drop out entirely.**
The only GRIB1 file in the pipeline is `lsm.grib`, and both things it supplies — the
`lsm == 0.0` mask and the `lsm` forcing channel — are `constant_in_time` and can be decoded
once offline into a `.npy`-equivalent (§10 A). `MaskValues` already works this way:
`mask.py:52` accepts a `.npy` path directly, and `:60-63` serialises whatever it loaded to
a temp `.npy` before use. So the runtime reader only ever sees the forecast files, which
are uniformly GRIB2 / `grid_ccsds`.

The `FieldSource` seam in §9 keeps this reversible: if inputs later narrow to one simple
encoding, dropping FFI is a trait impl rather than a rewrite.

### Where the input data comes from — `opendata` resolved

`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml` declares
`input: opendata`, and that input is **not implemented in anemoi-inference**:
`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/inputs/` has
no `opendata.py` at HEAD (`29dc717`) — it ships `mars`, `cds`, `gribfile`, `grib`,
`dataset`, `netcdf`, `fdb`, `opendap`, `cutout`, `split`, `repeated_dates`, `dummy`,
`empty`. It comes from a separate plugin, installed as
`anemoi-plugins-ecmwf-inference[opendata]` and discovered through the `input_registry`
(`inputs/__init__.py:18`); see
`/Users/sai/Documents/projects/anemoi/anemoi-inference/docs/usage/advanced/sources.rst:57-64`.

For a Rust port the practical consequence is that `opendata` carries no special semantics
worth replicating — it is a retrieval plugin over ECMWF's open data service. `mars`
(`inputs/mars.py`) and `cds` (`inputs/cds.py`) are the in-tree alternatives, both
requiring accounts but able to deliver N320 natively. The regridding question (open data
is 0.25° regular lat/lon; the model wants N320 reduced Gaussian) lives in that plugin, not
in anemoi-inference, so it must be settled separately before real-data inference works.

---

## 9. Proposed Rust architecture

All new files are under `/Users/sai/Documents/projects/airglow/aifsv2/src/`:

```
/Users/sai/Documents/projects/airglow/aifsv2/src/
  schema.rs                 ← SEAM 1: everything checkpoint-shaped, parsed at load
  input/
    mod.rs
    grib.rs                 eccodes → FieldMessage (values in stored order)
    source.rs               ← SEAM 2: trait FieldSource
    regrid.rs               0.25° regular_ll → N320, §8
    forcings.rs             §3 formulas, pure functions
    normalize.rs            affine transform + inverse
    assemble.rs             §5/§6 tensor construction
  graph/
    mod.rs
    load.rs                 mmap the extracted edge tensors — the default path
    builder.rs              OPTIONAL: kd-tree, cutoff + knn → edge_index [2, E]
    attributes.rs           OPTIONAL: haversine, Rodrigues directions, unit-std
```

`graph/builder.rs` extends the existing
`/Users/sai/Documents/projects/airglow/aifsv2/src/graph.rs`, which already defines
`GraphData`, its safetensors loader and `expand_edges`.

### Seam 1 — `/Users/sai/Documents/projects/airglow/aifsv2/src/schema.rs`

The 134 variable names, all six `data_indices` permutations, the per-variable provenance
flags, the bounding variable lists, the imputer fill lists and `multistep` live in
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/ai-models.json`.
Parse them at startup into:

```rust
pub struct Schema {
    pub variables: Vec<String>, // 134, canonical order
    pub multistep: usize,       // 2   = config.training.multistep_input
    pub timestep: Duration,     // 6h  = config.data.timestep
    pub input: IndexSet,        // full/prognostic/diagnostic/forcing, data space
    pub output: IndexSet,
    pub model_input: IndexSet, // model space, for the §6 residual and §11 roll
    pub model_output: IndexSet,
    pub var_to_input_channel: HashMap<String, usize>, // 134-space → 106-space
    pub output_channel_to_var: Vec<String>,           // 120-space → name
    pub computed_forcing: Vec<usize>,                 // `computed_forcing: true`  (9)
    pub constant_in_time: Vec<usize>,                 // `constant_in_time: true`  (9)
    pub boundings: Vec<Bounding>, // Relu | Hardtanh{min,max} | Fraction{total_var}
    pub imputer_zero: Vec<usize>,
    pub latitudes: Vec<f64>, // from anemoi-metadata/latitudes.numpy
    pub longitudes: Vec<f64>,
}
```

`anemoi-inference`'s `Metadata`
(`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/metadata.py`)
is a working reference for every one of these — mirror its derivations rather than
inventing them:

| field                     | accessor                                                            |
| ------------------------- | ------------------------------------------------------------------- |
| `multistep`               | `multi_step_input` `:336` → `config.training.multistep_input`       |
| `timestep`                | `timestep` `:235` → `config.data.timestep` = `6h`                   |
| 106                       | `number_of_input_features` `:325` = `len(indices.model.input.full)` |
| `var_to_input_channel`    | `variable_to_input_tensor_index` `:282`                             |
| `output_channel_to_var`   | `output_tensor_index_to_variable` `:302`                            |
| `model_input.prognostic`  | `prognostic_input_mask` `:353`                                      |
| `model_output.prognostic` | `prognostic_output_mask` `:348`                                     |
| `computed_forcing`        | `model_computed_variables` `:330`                                   |

The two channel maps are both built by `_make_indices_mapping(from, to)` (`:263`), which
zips `data.*.full` against `model.*.full` — that is the data-space ↔ model-space
translation §6 needs for the residual and §11 for the roll-back. Note also that
`prognostic_input_mask` / `prognostic_output_mask` **are** the arrays
`encoder_processor_decoder.py` calls `_internal_input_idx` / `_internal_output_idx`; same
two lists, two names.

`multi_step_output` is hardcoded to `1` for single-dataset checkpoints (`:341`), so a
single forward pass produces exactly one future step.

Hardcoding any of this as Rust constants would mean a recompile per checkpoint revision,
and would silently desync if a variable is added. Reading it makes a new checkpoint a data
change. This matters more than any other modularity decision here.

### Seam 2 — `FieldSource` (`/Users/sai/Documents/projects/airglow/aifsv2/src/input/source.rs`)

```rust
pub struct FieldMessage {
    pub param_id: u32,
    pub level: u32,
    pub level_type: LevelType, // Isobaric | Surface | HeightAboveGround | SoilLayer
    pub valid_time: DateTime<Utc>,
    pub grid: Grid,       // N320 | RegularLL { ni, nj, lat0, lon0, di, dj }
    pub values: Vec<f32>, // stored order; NaN at bitmapped points
}

pub trait FieldSource {
    fn fetch(&self, params: &[ParamSpec], times: &[DateTime<Utc>]) -> Result<Vec<FieldMessage>>;
}
```

`GribFiles` implements it now; a `SafetensorsFixture` implementation lets phases 1–4 below
run before the GRIB reader exists, and any future zarr/opendata path slots in without
touching `src/input/assemble.rs`.

`grid` is the one field an earlier draft of this document argued against — the reasoning
was that coordinates come from the checkpoint, so a source need not describe its grid.
That holds for `lsm.grib` and fails for open data, which arrives on `regular_ll` 1440×721
(§8). The message must carry enough geometry for `regrid.rs` to interpolate onto the
checkpoint's 542,080 points; it still never carries _coordinates_, only the handful of
section-3 keys that generate them. `Grid::N320` is the pass-through case.

Routing keys off `(param_id, level_type, level)` rather than a short name — see §8, where
six of the 97 inputs are not addressable by the model's variable name.

### Everything else is concrete

`src/input/forcings.rs`, `src/input/normalize.rs`, `src/input/assemble.rs` and `src/graph/`
(all under `/Users/sai/Documents/projects/airglow/aifsv2/`) have exactly one correct
behaviour each, fixed by the checkpoint. No traits, no config. Anemoi abstracts these
because it must support GNN processors, limited-area grids, ensembles and diffusion; none
of that applies.

---

## 10. Order of work

### First, three lists that are easy to conflate

The natural way to plan this is in data-flow order — read GRIB, regrid, pre-process,
assemble, run. That is the right description of the **runtime pipeline** and the wrong
build order, and it also tends to sweep the one-time work in with the per-step work. Three
separate things:

**A. Offline, once — extract, never implement.** Everything below is a file read plus a
reshape. After this session none of it requires an algorithm.

| from                         | what                                                              | replaces              |
| ---------------------------- | ----------------------------------------------------------------- | --------------------- |
| `ai-models.json`             | 134 names, six index sets, boundings, imputer lists, `multistep`  | §2, §4, §6            |
| `.ckpt` → `graph_data`       | `edge_index` ×2, `edge_length`, `edge_dirs` (already unit-std)    | **all of §7**         |
| `.safetensors`               | node attrs `[N,12]` ×2; normaliser `_norm_mul/_add/_idx`          | §4                    |
| `latitudes/longitudes.numpy` | f64 target coordinates                                            | grid geometry         |
| `lsm.grib`                   | the `lsm == 0.0` mask (542,080 bools) + the `lsm` forcing channel | GRIB1 + `grid_simple` |

That last row is worth stating plainly: baking the mask means the runtime reader never
sees a GRIB1 or `grid_simple` message, so its scope narrows to **GRIB2 + `grid_ccsds`**.

**B. Runtime, per step.** Note where the regrid sits — before the pre-processors, not
after. `apply-mask` compares `== 0.0` against a 542,080-element mask, and the imputer and
normaliser are per-channel over the model grid, so all three require N320 input.

```
GRIB2/ccsds decode → regrid 1440×721 → N320 → mwd→cos/sin, apply-mask
  → const imputer → normalise (affine) → assemble [542080, 224]
  → forward → prognostic residual → boundings → roll + write-back → repeat
```

**Boundings are output-side.** They belong at the end of this chain (§6), applied to the
120-channel output after the residual and in list order — not grouped with the forcings or
the input pre-processors, which they never share a tensor with.

**C. Build order — deliberately not B.** Data-flow order front-loads the GRIB reader and
the regridder: the two hardest components, and the two with no oracle until everything
downstream already works. The table below inverts that. Each phase is verifiable on its
own, with the oracle named. Paths are relative to
`/Users/sai/Documents/projects/airglow/aifsv2/`.

| # | file(s)                                   | oracle                                                                                                                                |
| - | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `src/schema.rs`, `src/input/normalize.rs` | round-trip a random vector through transform/inverse; parse all six index sets and check lengths 106/120/92/28/14                     |
| 2 | graph **extract** (offline script)        | shapes `[2, 748348]` / `[2, 1626240]`; `std(edge_length) == std(edge_dirs) == 1.0` on load — see §7, this replaces building           |
| 3 | `src/input/assemble.rs`                   | dump `x_data_latent` from Python, compare elementwise; check channels 0–105 vs 106–211 are the two timesteps                          |
| 4 | `src/aifs.rs`, `src/main.rs`              | wire to existing encoder/processor/decoder; compare against a Python `predict_step` dump                                              |
| 5 | `src/input/grib.rs`                       | `data/lsm.grib` — `scripts/parse_grib.py --stats` gives ground truth: 542,080 values, min 0.0, max 1.0, mean 0.2865346, std 0.4452423 |
| 6 | `src/input/forcings.rs`                   | compare against `earthkit` output for a fixed date; `insolation` ≥ 0 everywhere, = 0 on the night side                                |
| 7 | `src/input/regrid.rs`                     | regrid the 0.25° `lsm` from `20260810000000-0h-oper-fc.grib2` and compare against native `lsm.grib`; coastlines are the error term    |

Phases 1–4 need no GRIB reader at all — use a `SafetensorsFixture` source. Getting a
numerically-correct forward pass first makes phase 5 a swap rather than a debugging
session with two unknowns.

Phase 2 used to be the kd-tree build (`src/graph/builder.rs`, `src/graph/attributes.rs`)
and is now an extraction, per §7. Keep the builder as a **later** phase if you want it —
verified reproducible, and the right home for the 748,348 assertion as a regression test —
but it is no longer on the path to a working forward pass.

An unresolved blocker to settle before phase 5, because it changes what you retrieve:
**`sd` is absent from both sample GRIB files** (§8) and is a prognostic input, the
`ConditionalNaNPostprocessor` remap key, and one of the three `apply-mask` targets. Check
the open-data catalogue rather than building around the gap.

Phase 7 has a rare luxury: the same physical field exists on both grids, natively at N320
in `lsm.grib` and interpolated to 0.25° in the forecast file. That makes the regridder
checkable against ground truth instead of against itself. Expect disagreement at
coastlines regardless — the open-data copy is 8-bit packed, 129 distinct values against
63,747 — so compare distributions and interior points, not exact equality.

Phase 8 is §11: the autoregressive loop, which only becomes meaningful once 1–7 hold.

---

## 11. The autoregressive loop

§1–§10 get you one forward pass: 106 channels × 2 timesteps in, 120 channels out. A
forecast is that pass run N times, feeding each output back. The mechanics are in
`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/`.

### Shape chain per step

```
input tensor     (2, 106, 542080)             tensors.py:141-151
                                              (multi_step_input, features, values)
      ↓ swapaxes(-2,-1)[newaxis]              runner.py:443
                 (1, 2, 542080, 106)          = [batch, time, grid, vars] → §5
      ↓ predict_step
y_pred           (batch, [time], ensemble, values, variables)    runner.py:502
      ↓ squeeze(dim=(0,2))
outputs          (time, values, variables)    runner.py:508
      ↓ roll + prognostic write-back          tensors.py:412-417
      ↓ dynamic forcings for next_dates       tensors.py:434-478
next input tensor(1, 2, 542080, 106)
```

### Stepping

`forecast_stepper` (`runner.py:373-416`) derives the schedule:

```
output_horizon = timestep × multi_step_output       # 6h × 1 = 6h
steps          = ceil(lead_time / output_horizon)   # 10-day forecast → 40 steps
valid_dates[s] = start + s·output_horizon + timestep·(i+1)
```

With `multi_step_output = 1` (`metadata.py:341`) every model call advances the forecast by
one 6-hour timestep.

### Feeding the output back

`copy_prognostic_fields_to_input_tensor` (`tensors.py:392-431`):

```python
prognostic_fields = index_select(
    y_pred, dim=-1, index=prognostic_output_mask
)  # 92 of 120
keep_steps = min(multi_step_output, multi_step_input)  # 1
input_tensor = input_tensor.roll(-keep_steps, dims=1)  # :412
input_tensor[:, -1, :, prognostic_input_mask] = prognostic_fields[:, -1, ...]  # :415
```

The `roll` on the time axis shifts t−6h ← t; the write then fills the now-stale last row
with the prediction. **Only the 92 prognostic channels are carried over.** The 28
diagnostics are outputs only — they are never read back, which is why `model.input` has no
diagnostic entries.

Everything else in the 106 must be re-supplied before the next call:

- **Dynamic forcings** — recomputed for `next_dates` and written to the last row
  (`add_dynamic_forcings_to_input_tensor`, `tensors.py:434-478`). For this checkpoint that
  is the five time-dependent computed forcings from §2: `cos/sin_julian_day`,
  `cos/sin_local_time`, `insolation`. Note `load_forcings_array` returns
  `(variables, dates, values)` and the caller swaps to `(dates, variables, values)` before
  the scatter (`:456-460`).
- **Constant forcings** — `cos/sin_latitude`, `cos/sin_longitude`, `lsm`, `sdor`, `slor`,
  `z`, `wmb` are never rewritten. They survive the `roll` because rolling a 2-row tensor
  by 1 moves row 1 to row 0, and both rows already hold identical constants.

### The `check[]` invariant — port this

A bool array over the 106 input channels, maintained across the whole loop:

```
reset[i] = True  if variable i is constant_in_time     runner.py:461-466
check    = reset.copy()                                runner.py:468, re-armed :561
check[prognostic_input_mask] = True                    tensors.py:425  after roll-back
check[forcing.mask]          = True                    tensors.py:470  per forcing source
assert check.all()                                     runner.py:585-594
```

Writing a channel twice raises — `tensors.py:418-423` names the conflicting prognostic
variables, `tensors.py:469` is a bare `assert not check[source.mask].any()` for forcings.
Leaving one unwritten raises with the offending names (`runner.py:585-594`). This is the
single most valuable runtime guard in the whole pipeline: a missing forcing otherwise
produces a forecast that looks entirely plausible and is quietly wrong, and it will not
show up in a one-step test — only after several steps of drift.

Rust equivalent: a `[bool; 106]` (or a bitset) seeded from `schema.constant_in_time`,
asserted before each `forward`.

### State between steps

`anemoi-inference` carries a `State` — `dict` with `date`, `latitudes`, `longitudes`,
`step`, `previous_step`, and `fields: {name → array}` (`runner.py:556-565`,
`types.py:20`). The tensor is the model's view; the `State` is the human/output view, and
the field names are what post-processors and GRIB encoding key off.

For Rust, the tensor alone is sufficient to iterate — a `State` is only needed once you
want to emit output per step. Suggested minimum:

```rust
pub struct State {
    pub date: DateTime<Utc>,
    pub step: Duration,
    pub fields: HashMap<String, Vec<f32>>, // name → 542,080 values
}
```

with `latitudes`/`longitudes` held once in `Schema` rather than copied per step.

### Deferred

`accumulate_from_start_of_forecast` in
`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml` post-processes the six
accumulation variables (`cp`, `ro`, `sf`, `ssrd`, `strd`, `tp` — flagged
`process: "accumulation"` in §2) across steps; see
`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/post_processors/accumulate.py`.
Not needed to produce a correct prognostic rollout.

---

## Appendix — reproducing the numbers

```bash
source /Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/.venv/bin/activate

# tensor shapes quoted in §1
python /Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_safetensors.py -q latlons trainable lin_edge emb_nodes

# normaliser coefficient tensors (§4)
python /Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_safetensors.py -q processors

# GRIB geometry and value statistics (§8, phase 5)
python /Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_grib.py            # summary + grid block
python /Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_grib.py --stats    # stored values, not pygrib's expanded view
python /Users/sai/Documents/projects/airglow/aifsv2/scripts/parse_grib.py --json     # all 188 eccodes keys

# variable lists, data_indices, graph recipe, processors (§2, §4, §6, §7)
# note: the unwrapped copy — no nesting key needed
python -c "
import json
md = json.load(open('/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/ai-models.json'))
print(md['dataset']['variables'])
print(md['data_indices'])
print(json.dumps(md['config']['graph'], indent=2))
print(json.dumps(md['config']['data']['processors'], indent=2))
print(json.dumps(md['config']['model']['bounding'], indent=2))
"

# per-variable provenance flags (§2)
python -c "
import json
md = json.load(open('/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/ai-models.json'))
vm, vs = md['dataset']['variables_metadata'], md['dataset']['variables']
for flag in ('computed_forcing', 'constant_in_time'):
    print(flag, [v for v in vs if vm.get(v, {}).get(flag)])
print('mars       ', sum('mars' in vm.get(v, {}) for v in vs))
print('accumulation', [v for v in vs if vm.get(v, {}).get('process') == 'accumulation'])
"                                       # -> 9 / 9 / 125 / 6

# the f64 coordinate arrays (§1, §7) — raw little-endian f8, no .npy header
python -c "
import numpy as np
D = '/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata'
lat = np.fromfile(f'{D}/latitudes.numpy',  dtype='<f8')
lon = np.fromfile(f'{D}/longitudes.numpy', dtype='<f8')
print(lat.shape, len(np.unique(lat)), lat.min(), lat.max())   # (542080,) 640 ±89.78487690721863
print(lon[:3], lon.min(), lon.max())                          # [0. 20. 40.] 0.0 359.71875
"

# node coordinate layout claim (§1)
python -c "
import numpy as np; from safetensors import safe_open
with safe_open('/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0.safetensors', framework='numpy') as f:
    ll = f.get_slice('model.node_attributes.latlons_data')[:, :]
print(np.degrees(np.arctan2(ll[:,0], ll[:,2])).min(),   # latitude  → -89.784874
      np.degrees(np.arctan2(ll[:,0], ll[:,2])).max())   #           →  89.784874
"
```
