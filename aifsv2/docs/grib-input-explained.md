# GRIB, the three grids, and what still has to be built

This document explains the input side of AIFS-single 2.0 from first principles: what a GRIB
file is, what is inside the three GRIB files in
`/Users/sai/Documents/projects/airglow/aifsv2/data/`, the three different grids and tensor
shapes the data passes through, which of the model's variables are read out of GRIB and
which are computed, the exact anemoi call chain from a file on disk to a tensor entering the
model, and the gap between that chain and what
`/Users/sai/Documents/projects/airglow/aifsv2/src/` does today.

Its companion, `/Users/sai/Documents/projects/airglow/aifsv2/docs/grib-to-inference-pipeline.md`,
is the architecture spec: it decides _what to build in Rust and in what order_. This document
is the prerequisite — it establishes _what the data is_. Where the two touch, this one holds
the measurements and that one holds the decisions, and each points at the other rather than
repeating it.

Every number below was measured against the real files and against the anemoi working copies
at `/Users/sai/Documents/projects/anemoi/` (`anemoi-inference` at commit `29dc717`). The
appendix reproduces all of them.

---

## 1. What GRIB is

GRIB — GRIdded Binary, formally WMO FM-92 — is a container for **one two-dimensional field
per message**. A message is a self-describing, independently decodable byte range: it carries
its own grid definition, its own parameter identity, its own packing parameters and its own
payload. A GRIB _file_ is nothing more than those messages concatenated end to end.

```
  data/20260810000000-0h-oper-fc.grib2   (91,085,441 bytes, 126 messages)

  ┌──────────────┬──────────────┬──────────────┬─────┬──────────────┐
  │ message 1    │ message 2    │ message 3    │ ... │ message 126  │
  │ w @ 600 hPa  │ ...          │ ...          │     │ ...          │
  │ 1,260,246 B  │              │              │     │              │
  └──────────────┴──────────────┴──────────────┴─────┴──────────────┘
  ^0             ^1260246       ^...
  no file header.  no file trailer.  no table of contents.
```

There is no file-level header, no directory, and no shared dictionary between messages. Two
consequences follow, and both matter for a Rust reader:

1. **Concatenating two GRIB files is a valid GRIB file.** `cat a.grib b.grib > c.grib` is
   legal and is how multi-source retrievals are commonly assembled.
2. **Finding message _n_ means walking messages 1…_n−1_.** Every message begins with the
   ASCII bytes `GRIB` and, in edition 2, states its own total length at byte offset 8, so the
   walk is cheap — but it is a walk, not a lookup.

Point 2 has an escape hatch in this repository, covered in §2.4.

The authoritative references are the WMO Manual on Codes (FM-92 GRIB,
https://codes.wmo.int/) and ECMWF's ecCodes key documentation
(https://confluence.ecmwf.int/display/ECC/GRIB+Keys). Every key name quoted in this document
— `packingType`, `bitsPerValue`, `pl`, `scanningMode` — is an ecCodes key, which is a
computed view over the raw octets rather than a literal field in the spec.

### 1.1 Two editions, two section layouts

GRIB edition 1 and edition 2 are different formats that share a name and a magic number. Both
files types are present in `data/`, so both matter.

**Edition 2** has nine sections. Measured from message 1 of
`data/20260810000000-0h-oper-fc.grib2` (`w`, vertical velocity at 600 hPa):

```
section  bytes   what it holds                              answers
───────  ──────  ─────────────────────────────────────────  ─────────────────────
  0          16  "GRIB", discipline=0, edition=2,           is this GRIB2?
                 totalLength=1260246                        how long is this message?
  1          21  centre=ecmf, tablesVersion=36,             who made it, and when
                 dataDate=20260810, dataTime=0
  2          17  local use (ECMWF MARS keys)                stream/type/class
  3          72  gridDefinitionTemplate=0 (regular_ll),     WHERE are the values?
                 Ni=1440, Nj=721, scanningMode=0,
                 first=(90.0, 180.0), last=(-90.0, 179.75),
                 increments 0.25/0.25, shapeOfTheEarth=6
  4          34  productDefinitionTemplate=0,               WHAT is this variable?
                 parameterCategory=2, parameterNumber=8,
                 typeOfFirstFixedSurface=pl, value=60000,
                 forecastTime=0
  5          25  dataRepresentationTemplate=42 (CCSDS),     HOW are the bytes packed?
                 numberOfValues=1038240, bitsPerValue=16,
                 referenceValue=-10.7285509,
                 binaryScaleFactor=-12, decimalScaleFactor=0
  6           6  bitMapIndicator=255 (no bitmap)            which points are missing?
  7   1,260,051  the packed values                          the payload
  8           4  "7777"                                     end marker
───────  ──────
 total   1,260,246 = totalLength   ✓
```

Section 7 is 99.98% of the message. Everything else — all eight remaining sections — is 195
bytes of description.

**Edition 1** has six sections, numbered 0, 1, 2, 3, 4, 5, and sections 2 and 3 are optional.
Measured from `data/lsm.grib`:

```
section  bytes       what it holds
───────  ──────────  ────────────────────────────────────────────────────
  0               8  "GRIB", totalLength=1627628, edition=1
  1              52  table2Version=128, indicatorOfParameter=172 (lsm),
                     dataDate=20260311, level=0
  2           1,312  GRID DEFINITION: reduced_gg, N=320,
                     pl[640] (the per-row point counts), first=(89.785, 0.0),
                     last=(-89.785, 359.719)
  3               —  absent (bitmapPresent=0)
  4       1,626,252  BINARY DATA: grid_simple, bitsPerValue=24,
                     referenceValue=0.0, binaryScaleFactor=-23,
                     decimalScaleFactor=0
  5               4  "7777"
───────  ──────────
 total   1,627,628 = totalLength   ✓
```

Note the section numbering is not a renaming of the same content: edition 1's section 2 is the
grid, whereas edition 2's grid is section 3. Any hand-written parser must branch on the
edition before it can interpret a section number at all.

### 1.2 Packing: how bytes become floats

The packing section (5 in GRIB2, part of section 4 in GRIB1) declares a `packingType`. The two
that appear in `data/` are on opposite ends of the difficulty scale.

**`grid_simple`** — used by `data/lsm.grib`. Each value is an unsigned integer of
`bitsPerValue` bits, big-endian, packed without padding between values, and reconstructed by a
scale-and-shift:

```
    value[i] = referenceValue + raw[i] · 2^binaryScaleFactor · 10^(−decimalScaleFactor)

    for lsm.grib:  R = 0.0,  E = −23,  D = 0,  bitsPerValue = 24

    value[i] = raw[i] · 2^−23        (raw ∈ [0, 2^24 − 1] → value ∈ [0, ~2.0])
```

That is a few hundred lines of Rust including the section walker. The measured payload,
1,626,252 bytes for 542,080 values, is exactly `ceil(542080 × 24 / 8) = 1,626,240` plus 12
bytes of section-4 header — the arithmetic checks out, which is a useful sanity test for any
reader.

**`grid_ccsds`** (`dataRepresentationTemplateNumber = 42`) — used by **all 137 messages** in
the two forecast files. This is not a formula. It is CCSDS 121.0-B Adaptive Entropy Coding
(the Rice/Golomb-family codec implemented by libaec/szip), parameterised here by
`ccsdsFlags=14`, `ccsdsBlockSize=32`, `ccsdsRsi=128`. The scale-and-shift above still applies,
but only _after_ entropy decoding recovers the integers.

This single fact is what forces the eccodes dependency for the Rust port; the argument is made
in full in §8 of `grib-to-inference-pipeline.md` and is not repeated here.

### 1.3 Bitmaps

`bitMapIndicator = 255` means "no bitmap": every one of the `numberOfValues` positions carries
a real value. Anything else means a bitmap section is present, and the payload contains values
only for the set bits — decoded fields must have `missingValue` (or, more usefully, NaN)
inserted at the clear bits.

In `data/`, bitmaps appear on exactly 13 messages: the two `vsw` (soil moisture) messages in
the operational file — which carry `numberOfValues = 374,250` against
`numberOfDataPoints = 1,038,240`, so 64% of the globe is masked — and **all 11** wave
messages. That is not a coincidence — those are
precisely the fields that are physically undefined somewhere (soil moisture over ocean and
ice, wave parameters over land), and they are precisely the fields the checkpoint's
`ConstantImputer` fills with zero. Masked points must therefore arrive at the imputer as NaN;
a reader that silently zero-fills during unpacking destroys the distinction between "no soil
here" and "soil moisture is 0.0".

---

## 2. What is in the three files

### 2.1 `20260810000000-0h-oper-fc.grib2` — 126 messages, the bulk of the input

Every message agrees on geometry and provenance:

| key                                   | value                                   |
| ------------------------------------- | --------------------------------------- |
| `editionNumber`                       | 2 (all 126)                             |
| `gridType`                            | `regular_ll` (all 126)                  |
| `Ni` × `Nj`                           | 1440 × 721 = **1,038,240** points       |
| `packingType`                         | `grid_ccsds` (all 126)                  |
| `scanningMode`                        | 0 — i scans +ve, j scans −ve, row-major |
| first grid point                      | 90.0°N, **180.0°E**                     |
| last grid point                       | −90.0, 179.75                           |
| increments                            | 0.25° × 0.25°                           |
| `shapeOfTheEarth`                     | 6 (sphere, radius 6,371,229 m)          |
| `dataDate` / `dataTime` / `stepRange` | 20260810 / 0 / **0**                    |
| `marsStream` / `marsType`             | `oper` / `fc`                           |

and disagree on only two things: `bitsPerValue` (16 on 42 messages, 12 on 77, 8 on 1, and
**0 on 6**) and `bitmapPresent` (1 on the two `vsw` messages, 0 on the other 124).

`bitsPerValue = 0` is a degenerate but legal encoding meaning _the field is constant_: every
value equals `referenceValue` and section 7 carries no data. Measured, it lands on exactly the
six accumulation variables — `ssrd`, `strd`, `cp`, `sf`, `tp`, `rowe` — which at step 0 are
identically zero. A reader that assumes at least one bit per value will divide by zero or read
past the section. All six are diagnostic-only, so they can equally be dropped on input.

The single 8-bit message is `lsm`, which is why the open-data land-sea mask resolves to only
129 distinct values (§2.3).

The full parameter inventory:

```
shortName  paramId   typeOfLevel        n   levels
─────────  ────────  ─────────────────  ──  ─────────────────────────────────────────
t             130    isobaricInhPa      14  10 50 100 150 200 250 300 400 500 600 700 850 925 1000
u             131    isobaricInhPa      14  (same 14)
v             132    isobaricInhPa      14  (same 14)
z             129    isobaricInhPa      14  (same 14)
w             135    isobaricInhPa      14  (same 14)
gh            156    isobaricInhPa      14  (same 14)      <- redundant with z
q             133    isobaricInhPa      13  50 … 1000  (no 10 hPa)
─────────                               97  = pressure-level messages
2t            167    heightAboveGround   1  2
2d            168    heightAboveGround   1  2
10u           165    heightAboveGround   1  10
10v           166    heightAboveGround   1  10
100u       228246    heightAboveGround   1  100
100v       228247    heightAboveGround   1  100
sot        260360    soilLayer           2  layers 1, 2     <- stl1, stl2
vsw        260199    soilLayer           2  layers 1, 2     <- swvl1, swvl2  (bitmapped)
msl           151    meanSea             1
tcc        228164    entireAtmosphere    1
tcw           136    entireAtmosphere    1
lcc          3073    lowCloudLayer       1
mcc          3074    mediumCloudLayer    1
hcc          3075    highCloudLayer      1
sp            134    surface             1
skt           235    surface             1
z             129    surface             1                  <- surface geopotential
lsm           172    surface             1
sdor          160    surface             1
slor          163    surface             1
fscov      260289    surface             1                  <- snowc, output encoding only
cp         228143    surface             1   bitsPerValue=0
sf         228144    surface             1   bitsPerValue=0
tp         228228    surface             1   bitsPerValue=0
ssrd          169    surface             1   bitsPerValue=0
strd          175    surface             1   bitsPerValue=0
rowe       231002    surface             1   bitsPerValue=0
─────────                               29
                                       126  total
```

### 2.2 `20260810000000-0h-wave-fc.grib2` — 11 messages

Same 1440 × 721 grid, same `grid_ccsds`, `bitsPerValue = 16` throughout, `marsStream = wave`,
and — unlike the operational file — **`bitmapPresent = 1` on every message**, because wave
parameters are undefined over land.

```
#1   h1012   140114  surface           wave height, 10–12 s period band
#2   h1214   140115  surface
#3   h1417   140116  surface
#4   h1721   140117  surface
#5   h2125   140118  surface
#6   h2530   140119  surface
#7   wmb     140219  surface           model bathymetry     <- a *forcing*
#8   swh     140229  surface           significant wave height
#9   mwd     140230  surface           mean wave direction  <- becomes cos_mwd/sin_mwd
#10  mwp     140232  surface           mean wave period
#11  cdww    140233  heightAboveSea=10 drag coefficient
```

### 2.3 `lsm.grib` — 1 message, and the only one on the model's own grid

```
editionNumber        1                    <- GRIB1, unlike everything else
gridType             reduced_gg           <- the N320 grid the model runs on
N                    320
pl                   640 entries, summing to 542,080
packingType          grid_simple
bitsPerValue         24                   <- against 8–16 in the open-data files
paramId              172 (lsm)
first / last point   (89.785, 0.0) / (−89.785, 359.719)
dataDate             20260311
```

This file is not part of the same retrieval as the other two. It comes from MARS at native
resolution, upstream of the interpolation that produces open data, and it exists in this
repository for two reasons at once: it is the `lsm` forcing channel, and it is the mask source
for the `apply-mask` pre-processor on `sd`, `swvl1` and `swvl2`
(`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:10-17`).

The resolution difference is measurable rather than theoretical: the native `lsm` here has
**63,747 distinct values**, against **129** for the 8-bit open-data copy of the same field.
Land fraction interpolated across a coastline and then quantised to 129 levels is a materially
different field.

### 2.4 The `.index` sidecars — a message directory the format does not provide

`data/20260810000000-0h-oper-fc.index` and its wave counterpart are **not** GRIB. They are
JSON-lines files written by the ECMWF open-data client, one record per message, carrying MARS
keys plus a byte range:

```json
{
  "domain": "g",
  "date": "20260810",
  "time": "0000",
  "expver": "0001",
  "class": "ai",
  "type": "fc",
  "stream": "oper",
  "step": "0",
  "levelist": "600",
  "levtype": "pl",
  "param": "w",
  "model": "aifs-single",
  "_offset": 0,
  "_length": 1260246
}
```

Measured: the 126 records are **contiguous** (`_offset[i] + _length[i] == _offset[i+1]` for
every _i_) and their lengths sum to exactly 91,085,441 bytes — the full file. The same holds
for the 11 wave records against 8,694,454 bytes.

This means the byte range of any message is addressable by `(param, levtype, levelist)`
without parsing a single GRIB octet. For a Rust reader that needs 97 of 137 messages, that
converts "walk 126 message headers, decode, discard 45" into "seek, read `_length` bytes,
decode exactly what is needed". It is not part of the format and it will not exist for a MARS
retrieval, so it belongs behind the same source abstraction as everything else — but for the
sample data on disk it is free.

---

## 3. Three grids, three shapes

### 3.1 Grid A — the 0.25° regular lat/lon mesh

This is what the two `.grib2` files carry. It is a plain raster: 721 rows by 1440 columns,
row-major, north to south, and — this is the trap — **starting at the dateline, not at
Greenwich**.

```
                    i = 0        1        2      ...    719      720    ...   1439
                  lon=180.0   180.25   180.5         359.75      0.0        179.75
                    │          │        │              │          │           │
   j=0   lat= 90.00 ●──────────●────────●── ...  ──────●──────────●── ... ────●
   j=1   lat= 89.75 ●──────────●────────●── ...  ──────●──────────●── ... ────●
   j=2   lat= 89.50 ●──────────●────────●── ...  ──────●──────────●── ... ────●
          ...
   j=360  lat= 0.00 ●──────────●────────●── ...  ──────●──────────●── ... ────●
          ...
   j=720  lat=-90.00 ●─────────●────────●── ...  ──────●──────────●── ... ────●

   flat index   k = j·1440 + i          (scanningMode = 0)
   latitude     lat_j = 90 − 0.25·j     j ∈ [0, 720]
   longitude    lon_i = (180 + 0.25·i) mod 360     i ∈ [0, 1439]

   total 721 × 1440 = 1,038,240 points
```

Both poles are represented as full 1440-point rows of identical values, and both the 180°
meridian and 0° meridian appear once. The rows are equally spaced in latitude, which is what
makes this a _regular_ grid and what makes it **not** the grid the model was trained on.

### 3.2 Grid B — N320 reduced Gaussian, the model grid

The model runs on 542,080 points arranged as a _reduced Gaussian_ grid. Two things distinguish
it from Grid A.

First, the latitude rows are not equally spaced. They are the 640 roots of the Legendre
polynomial P₆₄₀, running from ±89.78487690721863° at the poleward rows. The departure from
uniformity is small but real: measured row spacing runs from 0.278674° at the poles to
0.281030° in mid-latitudes and the tropics, so the poleward rows sit marginally closer
together. It is not enough to see in a diagram, and it is more than enough to make
"latitude = 90 − k·Δ" wrong.

Second, and more visibly, **each row has a different number of points**, chosen to keep the
physical spacing roughly constant instead of letting points crowd together near the poles.
Those counts are the `pl` array, and they are measured to be symmetric about the equator:

```
   pl[0..12]  = [18, 25, 36, 40, 45, 50, 60, 64, 72, 72, 75, 81, ...]
   pl[-12..]  = [81, 75, 72, 72, 64, 60, 50, 45, 40, 36, 25, 18]
   sum(pl)    = 542,080        symmetric: pl == reversed(pl)
   max(pl)    = 1280, held flat across rows 246…393 — 148 rows, |lat| ≤ 20.66°


   row   0  lat  89.785   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●             18 points
   row   1  lat  89.506   ●  ●  ●  ●  ●  ●  ●  ●  ● ... ●                 25
   row   2  lat  89.226   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ... ●             36
    ...
   row 246  lat  20.656   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● ... ●       1280  ┐
    ...                                                                       │ plateau
   row 319  lat   0.141   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● ... ●       1280  │ 148 rows
   row 320  lat  -0.141   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● ... ●       1280  │
    ...                                                                       │
   row 393  lat -20.656   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● ... ●       1280  ┘
    ...
   row 639  lat -89.785   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●             18

   flat index of row r, column c:   k = (Σ_{m<r} pl[m]) + c
   longitude within row r:          lon = c · 360 / pl[r]        c ∈ [0, pl[r])

   For comparison, the *full* Gaussian grid at the same truncation would be
   640 × 1280 = 819,200 points.  The reduction saves 277,120 points, ~34%.
```

Note that the point count does not peak at the equator and fall away smoothly — it rises to
1280 by row 246 and then stays there for the whole tropical band before falling again.

The row-offset arithmetic is worth stating explicitly because it is the only thing standing
between a flat 542,080-element array and a geolocated field. Measured: row 0 starts at index
0, row 1 at 18, row 319 at 269,760, row 320 at 271,040.

**The coordinates are not in the file, and this pipeline never computes them.** Section 2 of
`lsm.grib` is 1,312 bytes — `N` plus the 640-entry `pl` — describing 542,080 points. Turning
that into coordinates requires Newton-iterating the roots of P₆₄₀, which is the substance of
what ecCodes provides. But the checkpoint already ships the answer as raw little-endian f64,
no `.npy` header, 542,080 × 8 bytes each:

```
data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/latitudes.numpy
data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/longitudes.numpy
```

and `/Users/sai/Documents/projects/airglow/aifsv2/src/metadata.rs:300` already reads them
(`read_f64_array`). A GRIB reader for this pipeline therefore has to produce **values in
stored order and nothing else**; value _i_ is node _i_.

### 3.3 A → B is a mandatory regrid, and it is a sparse matrix multiply

Grid A has 1,038,240 points. Grid B has 542,080. They are different grids with different
latitude spacings and no shared point layout, so every field retrieved from open data must be
interpolated before it can enter the model.

The important discovery is that earthkit does not _compute_ an interpolation — it applies a
precomputed sparse operator. `earthkit/regrid/interpolate.py:30-41` is, in full:

```python
def _interpolate(values, in_grid, out_grid, method, **kwargs):
    z, shape = find(in_grid, out_grid, method, **kwargs)
    if z is None:
        raise ValueError(f"No matrix found! {in_grid=} {out_grid=} {method=}")
    values = values.reshape(-1, 1)
    values = z @ values
    return values.reshape(shape)
```

`z` is a `scipy.sparse` matrix loaded from a `.npz`
(`earthkit/regrid/db.py:15`, `:376`), fetched on demand from ECMWF's matrix repository at
`https://sites.ecmwf.int/repository/earthkit/regrid/db/1/` (`db.py:25`) and cached locally.
For the A → B case `z` is `[542080, 1038240]`, and applying it is one sparse matrix–vector
product per field.

For the Rust port this collapses a large task into a small one: obtain and cache one matrix
once, then implement a CSR spmv. There is no interpolation kernel, no stencil search, and no
Gaussian-latitude solver on the runtime path.

(The retrieval plugin that performs this step for `input: opendata` is not in the
anemoi-inference tree; see §5.3.)

### 3.4 The internalised shapes, end to end

```
┌─ GRIB message, values in stored order ────────────────────────────────────┐
│  (1_038_240,)   f32, grid A, NaN at bitmapped points                      │
└───────────────────────────────────────────────────────────────────────────┘
            │  regrid:  z @ v          earthkit/regrid/interpolate.py:38
            ▼
┌─ one variable, one valid time ────────────────────────────────────────────┐
│  (542_080,)     f32, grid B                                               │
└───────────────────────────────────────────────────────────────────────────┘
            │  stack the 2 valid times   inputs/ekd.py:328-341
            ▼
┌─ state["fields"][name] ───────────────────────────────────────────────────┐
│  (2, 542_080)   f32     dates = [t−6h, t]                                 │
└───────────────────────────────────────────────────────────────────────────┘
            │  allocate NaN, scatter each named field to its channel
            │                            tensors.py:143-166
            ▼
┌─ input_tensor_numpy ──────────────────────────────────────────────────────┐
│  (2, 106, 542_080)      (multi_step_input, features, values)              │
│  = 460 MB at f32                                                          │
└───────────────────────────────────────────────────────────────────────────┘
            │  swapaxes(-2, -1)[newaxis]  runner.py:443
            ▼
┌─ x, what predict_step receives ───────────────────────────────────────────┐
│  (1, 2, 542_080, 106)   [batch, time, grid, vars]                         │
└───────────────────────────────────────────────────────────────────────────┘
            │  rearrange "b t e grid vars -> (b e grid) (time vars)"
            │  cat node_attributes [542_080, 12]
            │                            encoder_processor_decoder.py:198-204
            │                            src/aifs.rs:155-183
            ▼
┌─ x_data_latent, the encoder's source ─────────────────────────────────────┐
│  (542_080, 224)   224 = 2 × 106 + 12                                      │
│  = 486 MB at f32, live for the whole forward pass                         │
└───────────────────────────────────────────────────────────────────────────┘
```

Two properties of that last tensor are load-bearing and neither is checked by any shape
assertion:

**Time is the outer index of the 224.** The `einops` pattern is `(time vars)`, so channel
`i·106 + j` is variable _j_ at timestep _i_:

```
┌──────────────────────────┬──────────────────────────┬─────────────────┐
│ channels   0 … 105       │ channels 106 … 211       │ channels 212…223│
│ 106 variables at t − 6h  │ 106 variables at t       │ node attributes │
└──────────────────────────┴──────────────────────────┴─────────────────┘
                                                        │
                      [ sin_lat, sin_lon, cos_lat, cos_lon | trainable_0…7 ]
                        ── 4 read from the checkpoint ──   ── 8 learned ──
```

Interleaving the two timesteps instead produces a tensor of exactly the right shape that
silently yields a wrong forecast.

**The last 12 channels have nothing to do with GRIB.** All twelve are read from the
checkpoint: four encode where the node is (as sin/cos of latitude and longitude, not raw
degrees), and eight are a learned per-node embedding. A forecast changes 184 of the 224
channels — 92 prognostics × 2 timesteps — and never these.

The output side is narrower:

```
x_out   (542_080, 120)      120 = 92 prognostic + 28 diagnostic
          │  prognostic residual, x_out[…, out_idx] += x_skip[…, in_idx]
          │                         encoder_processor_decoder.py:222
          │                         src/aifs.rs:191-202
          │  boundings, in list order  (not yet implemented in src/)
          ▼
the forecast at t + 6h
```

---

## 4. Prognostic, diagnostic, forcing — and where each comes from

### 4.1 The two classifications, which are orthogonal

The checkpoint classifies its 134 variables twice, along axes that are easy to conflate:

- **Role** (`data_indices`) says how the model _uses_ a variable — prognostic, diagnostic or
  forcing.
- **Provenance** (`dataset.variables_metadata`) says where a variable _comes from_ — computed,
  constant in time, or retrieved with a specific MARS request.

Both are declared in
`/Users/sai/Documents/projects/airglow/aifsv2/data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/ai-models.json`.
Neither should be inferred from variable names.

### 4.2 Roles: 134 dataset variables → 106 in, 120 out

Measured counts:

| index space | tensor | full    | prognostic | diagnostic | forcing |
| ----------- | ------ | ------- | ---------- | ---------- | ------- |
| `data`      | input  | **106** | 92         | 28         | 14      |
| `data`      | output | **120** | 92         | 28         | 14      |
| `model`     | input  | **106** | 92         | **0**      | 14      |
| `model`     | output | **120** | 92         | 28         | **0**   |

The `data` rows number variables by their position in the 134-long dataset list; the `model`
rows number them by their channel in the actual tensor. The `data` counts do not sum to
`full` — `data.input` lists 28 diagnostics that the input tensor does not carry — which is
exactly why the `model` space exists as a separate numbering.

```
                        ┌───────────────────────────────────────────┐
                        │      134 dataset variables                │
                        └───────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
     92 PROGNOSTIC                 28 DIAGNOSTIC                 14 FORCING
     in AND out                    out only                     in only
          │                             │                             │
          ├──────────────┐              │                             │
          ▼              ▼              ▼                             ▼
   ┌────────────────────────┐   ┌────────────────────────┐
   │  model INPUT   106     │   │  model OUTPUT   120    │
   │  92 prognostic         │   │  92 prognostic         │
   │  14 forcing            │   │  28 diagnostic         │
   │   0 diagnostic         │   │   0 forcing            │
   └────────────────────────┘   └────────────────────────┘

   forcings are fed in and never predicted;  diagnostics are predicted and never fed in.
```

**The 92 prognostics** — the state the forecast actually carries forward. 24 surface and
single-level fields:

```
10u  10v  2d  2t  cdww  cos_mwd  sin_mwd  h1012  h1214  h1417  h1721  h2125  h2530
msl  mwp  sd  skt  sp  stl1  stl2  swh  swvl1  swvl2  tcw
```

plus 68 upper-air fields on the level set
`{10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}` hPa:

```
t_*  14 levels        u_*  14 levels        v_*  14 levels        z_*  14 levels
q_*  12 levels  (no q_10 at all; q_50 is diagnostic, not prognostic)
w_*   0 levels  (entirely diagnostic)
                                                          24 + 56 + 12 = 92  ✓
```

Note that `z` (surface geopotential, a _forcing_) and `z_10 … z_1000` (upper-air geopotential,
_prognostic_) are different variables that share a stem.

**The 28 diagnostics** — predicted, never fed back:

```
100u  100v  cp  hcc  lcc  mcc  q_50  ro  sf  snowc  ssrd  strd  tcc  tp
w_10  w_50  w_100  w_150  w_200  w_250  w_300  w_400  w_500  w_600  w_700
w_850  w_925  w_1000
```

`data_indices.model.input.diagnostic` is empty, so there is nothing to source for any of them.
The 28 diagnostic messages present in the GRIB files are surplus on input.

**The 14 forcings** — fed in, never predicted:

```
cos_julian_day  cos_latitude  cos_local_time  cos_longitude  insolation
sin_julian_day  sin_latitude  sin_local_time  sin_longitude
lsm  sdor  slor  wmb  z
```

### 4.3 Provenance: 97 retrieved, 9 computed

The 14 forcings split cleanly along the provenance axis, and that split is the whole answer to
"which ones come from GRIB":

```
  14 FORCINGS
  │
  ├── 9 with computed_forcing: true      ── pure functions, no file, no network
  │   │
  │   ├── 4 also constant_in_time: true  ── cos_latitude   sin_latitude
  │   │   compute once for the grid,        cos_longitude  sin_longitude
  │   │   never again
  │   │
  │   └── 5 time-dependent               ── cos_julian_day  sin_julian_day
  │       recompute at every step           cos_local_time  sin_local_time
  │                                         insolation
  │
  └── 5 retrieved, all constant_in_time  ── lsm  sdor  slor  wmb  z
      fetch once, reuse for every step

  ⇒ of the 106 input channels:  97 retrieved from GRIB  +  9 computed
                                 (= 92 prognostic + 5 retrieved forcings)
```

The `constant_in_time` flag holds exactly 9 variables — the four position forcings plus the
five retrieved ones — and it is what the rollout uses to decide which channels survive the
time-axis roll untouched.

One inconsistency worth knowing: `/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:23-25`
declares `constant_fields: [z, sdor, slor, lsm]`, omitting `wmb`, while the checkpoint flags
`wmb` as `constant_in_time`. The checkpoint flag is authoritative.

### 4.4 Where the 9 computed forcings are computed in anemoi

They are not computed in anemoi at all, strictly speaking — anemoi delegates to earthkit, and
uses the _same_ code path at dataset-build time and at inference time.

At inference, `ComputedForcings.load_forcings_array`
(`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/forcings.py:139-160`)
calls `ekd.from_source("forcings", source, date=dates, param=self.variables)` over an
`UnstructuredGridFieldList` built from the current state's latitudes and longitudes. That
source is `ForcingMaker` in
`.../site-packages/earthkit/data/sources/forcings.py`, and the returned array is ordered
`(variable, date, values)` (`forcings.py:177`).

The consequence for the Rust port is that one implementation of these nine formulas matches
training and inference simultaneously. The formulas themselves — including the truncated
Fourier series for solar declination, the day-of-year definition, and the three different
longitude unit conventions in play — are transcribed in §3 of
`grib-to-inference-pipeline.md`.

### 4.5 Which of the 97 are actually present in `data/`

Joining the 97 retrievable input variables against every message in the three files:

```
97 variables that must be retrieved
│
├── 90  matched directly on (shortName, level)
│       includes all 68 upper-air prognostics and 22 surface fields
│
├──  6  present but NOT addressable by the model's variable name:
│       stl1, stl2      ←  sot  @ soilLayer 1, 2      paramId 260360
│       swvl1, swvl2    ←  vsw  @ soilLayer 1, 2      paramId 260199
│       cos_mwd, sin_mwd ← mwd                         paramId 140230  (wave file)
│
└──  1  ABSENT from both files:   sd  (snow depth)
```

Two design consequences.

**Route on `(paramId, typeOfLevel, level)`, not on short name.** This is why
`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:27` sets
`use_grib_paramid: true`, and why a `LevelType` enum has to cover `soilLayer` and
`heightAboveSea` alongside `isobaricInhPa` and `surface`.

**`cos_mwd`/`sin_mwd` are a transform, not a retrieval.** A single `mwd` field becomes two
channels through the `cos_sin_mean_wave_direction` filter
(`.../site-packages/anemoi/transform/filters/fields/cos_sin_mean_wave_direction.py:22-51`),
wired in at `/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:9`. Its
`backward` direction reassembles `mwd` from the pair on output. This is the correct way to
handle a circular quantity: interpolating or averaging degrees across the 0/360 discontinuity
is meaningless.

**`sd` is an open blocker.** It is a prognostic input, the `remap` key for
`ConditionalNaNPostprocessor`, and one of the three `apply-mask` targets — not droppable.
Either the download's parameter list was short or open data does not serve it at 0.25°; the
catalogue settles it, not the code.

Separately, 45 of the 137 messages are surplus on input: the 28 diagnostics plus 14 `gh`
fields redundant with `z` on pressure levels, plus `fscov` (which
`inference.yaml:35-39` maps to `snowc` purely so the _output_ GRIB encoding has a template).

And a timing gap: every message in both files is `stepRange = 0`, so only one valid time is on
disk. `multistep = 2` needs a second retrieval at t − 6h.

### 4.6 What `src/` already knows about all this

`/Users/sai/Documents/projects/airglow/aifsv2/src/metadata.rs` parses the whole classification
at startup and is, as far as roles and provenance go, complete:

| concept in this section          | field in `Metadata`                                        | line    |
| -------------------------------- | ---------------------------------------------------------- | ------- |
| the 134 names, canonical order   | `variables`                                                | 74      |
| the four index sets              | `data_input`, `data_output`, `model_input`, `model_output` | 91-94   |
| 134-space → 106-space            | `var_to_input_channel`                                     | 96      |
| 120-space → name                 | `output_channel_to_var`                                    | 98      |
| the 9 computed forcings          | `computed_forcing`                                         | 100     |
| the 9 constant-in-time           | `constant_in_time`                                         | 101     |
| the 17 imputer-zero variables    | `imputer_zero`                                             | 102     |
| the 4 boundings, in order        | `boundings`                                                | 103     |
| the 542,080 f64 coordinates      | `latitudes`, `longitudes`                                  | 106-107 |
| `multistep = 2`, `timestep = 6h` | `multistep`, `timestep`                                    | 75-76   |

What is _not_ there is any consumer: nothing reads `computed_forcing` to compute anything,
nothing reads `imputer_zero` to impute anything, and nothing maps a variable name to a GRIB
message. §6 enumerates that gap.

---

## 5. The anemoi code path, GRIB file to model

All paths in this section are relative to
`/Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/` at `29dc717`.

### 5.1 File on disk → `State`

```
inputs/gribfile.py:16-21     @input_registry.register("grib")
                             GribFileInput(FieldlistInput, GribInput)
                             patterns = ("*.grib", "*.grb", "*.grb2", "*.grib2")
        │
inputs/ekd.py:580-614          @cached_property _fieldlist
                                 ekd.from_source("file", path)        <- earthkit opens it
                                 (also handles globs and directories)
        │
inputs/ekd.py:539-556          create_input_state(date=…)
        │
inputs/ekd.py:383-446            _create_input_state
        :435                       dates = [date + h for h in metadata.lagged]
                                   metadata.py:227-233  lagged = sorted([-s·timestep
                                     for s in range(multi_step_input)])
                                   = [-6h, 0h]  →  dates = [t−6h, t]
        │
inputs/ekd.py:238-381            _create_state                        <- the important one
        :283-307                   latitudes/longitudes: from fields[0].grid_points(),
                                     falling back to the checkpoint's supporting arrays
        :309                       state = {date, latitudes, longitudes, fields=FieldList}
        :312                       state = self.pre_process(state)
                                     ▲ fields is still a FieldList here, NOT numpy —
                                       this is where the two YAML pre_processors run:
                                       pre_processors/forward_transform_filter.py:68
                                         → cos_sin_mean_wave_direction  (mwd → cos/sin)
                                       pre_processors/mask.py:50-57
                                         → apply-mask on sd/swvl1/swvl2 using lsm.grib
        :323                       _filter_and_sort  (:163-216)
                                     sel(valid_datetime=…), sel(name=self.variables),
                                     order_by(name, valid_datetime ascending)
        :328                       n_points = fields[0].to_numpy(flatten=True).size
        :332-341                   allocate (len(dates), n_points) NaN per name,
                                     fill by date index
        :360                       state["fields"] : dict[name → (2, 542_080)]
        :362-367                   raise if any variable is missing a date
        :376                       set_private_attributes → GRIB templates for output
                                     (inputs/grib.py:24-43)
```

Note the ordering at `:312` versus `:323`: **pre-processors see a `FieldList`, not an array**,
and they run _before_ variable selection. `MaskValues` therefore compares against a
542,080-element mask and needs its input already on grid B — which is why the regrid has to
happen inside the retrieval plugin, upstream of everything here.

`MaskValues.__init__` (`pre_processors/mask.py:50-57`) resolves its mask in three branches: a
checkpoint supporting array, a `.npy` path, or — the branch this configuration takes —
`ekd.from_source("file", mask)[0].to_numpy(flatten=True)`. **No coordinates are read from
`lsm.grib` at all**, only values in stored order.

### 5.2 `State` → tensor → model

```
tensors.py:143-151           input_tensor_numpy = np.full(
                               (multi_step_input, number_of_input_features, n_values),
                               np.nan, dtype=np.float32)
                             = (2, 106, 542_080)
        │
tensors.py:157-166             for var, field in input_fields.items():
                                 i = variable_to_input_tensor_index[var]
                                 input_tensor_numpy[:, i] = field
        │
tensors.py:168-172             assert every channel was written, naming the missing
                               variables if not  ← the NaN fill exists for this check
        │
runner.py:443                torch.from_numpy(
                               np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...])
                             (2, 106, 542_080) → (2, 542_080, 106) → (1, 2, 542_080, 106)
        │
runner.py:339-371            predict_step(model, tensors, fcstep=…, step=…, date=…)
        │
runner.py:493                  y_pred = self.predict_step(…)
runner.py:508                  torch.squeeze(tensor, dim=(0, 2))
                                 → (time, values, variables)
```

and inside the model
(`/Users/sai/Documents/projects/anemoi/anemoi-core/models/src/anemoi/models/models/encoder_processor_decoder.py`):

```
:186-207   _assemble_input
  :187       x_skip = x[:, -1, ...]                    last timestep, for the residual
  :189       _apply_truncation(x_skip)                 NO-OP on this checkpoint
  :198-204   cat([rearrange(x, "b t e grid vars -> (b e grid) (time vars)"),
                  node_attributes_data])               → [542_080, 224]
:362       x_hidden_latent = node_attributes(hidden)   → [40_320, 12]
:366       encoder                                     → x_latent [40_320, 1024]
:386       x_latent_proc = processor(x_latent) + x_latent      (latent_skip: true)
:391       decoder(src=x_latent_proc, dst=x_data_latent)       → [542_080, 120]
:209-227   _assemble_output
  :222       x_out[…, out_prognostic] += x_skip[…, in_prognostic]
  :224       boundings, in list order
```

### 5.3 Where `SimpleRunner` sits, and what it does not do

`runners/simple.py` is the **low-level API**. Rather than driving retrieval itself, the caller
constructs a `State` and drives the `run` generator directly (`:75-102`). Its docstring
(`:47-51`) states the contract:

> The input State must contain all fields expected by the model, except for computed forcings
> which will be loaded by the runner. The user is responsible for providing any constant and
> dynamic forcings in the input State and during rollout.

That is enforced by `SimpleTensorHandler` (`:28-41`), which overrides both
`create_constant_coupled_forcings` and `create_dynamic_coupled_forcings` to return `[]` with a
warning. In terms of §4.3: **`SimpleRunner` fills in the 9 computed forcings and nothing
else.** The 92 prognostics and the 5 retrieved forcings must already be in the `State`, on
grid B, for both valid times. `execute()` raises `NotImplementedError` by design (`:71-74`);
`SimpleRunner` is not a command-line entry point.

This is the narrower of the two contracts and the more useful one to target from Rust: it
draws the boundary in exactly the place a port wants it, at "here is a fully assembled input".

One loose end in the configuration.
`/Users/sai/Documents/projects/airglow/aifsv2/data/inference.yaml:6` declares
`input: opendata`, and **there is no `opendata.py` in `inputs/`** at `29dc717` — the tree ships
`mars`, `cds`, `gribfile`, `grib`, `dataset`, `netcdf`, `fdb`, `opendap`, `cutout`, `split`,
`repeated_dates`, `dummy` and `empty`. It is a separate distribution,
`anemoi-plugins-ecmwf-inference[opendata]`, discovered through `input_registry`
(`inputs/__init__.py:18`); there is no `anemoi.plugins` package in the local venv either. The
regrid of §3.3 lives in that plugin, which is why it has no home anywhere in the chain above.

---

## 6. Where `src/` stands, and what is left to build

### 6.1 Done

| concern                                                                                    | where                                |
| ------------------------------------------------------------------------------------------ | ------------------------------------ |
| checkpoint schema: names, four index sets, channel maps, flags, boundings, f64 coordinates | `src/metadata.rs:110-173`            |
| the graph: two `edge_index`, `edge_dirs`, `edge_length`, node `x`                          | `src/graph.rs`                       |
| node attributes `[N, 12]`, both node sets                                                  | `src/named_node_attributes.rs`       |
| encoder / processor / decoder, and the 268-tensor weight load                              | `src/aifs.rs`, `src/main.rs:76-132`  |
| value-level verification that weights arrived untransposed                                 | `src/main.rs:143-190` (`spot_check`) |
| `_assemble_input` → `[grid, 224]`, correct `(time vars)` order                             | `src/aifs.rs:155-183`                |
| the prognostic residual across the two index spaces                                        | `src/aifs.rs:191-202`                |
| an end-to-end forward at synthetic size                                                    | `src/main.rs:196-239`                |

The model half is finished. The input half does not exist: `src/main.rs:211-219` still feeds
the model `Tensor::zeros`, and `Cargo.toml` has no GRIB, HTTP or sparse-linear-algebra
dependency.

### 6.2 Missing — the path to a real `[1, 2, 542080, 106]` in memory

Ordered by position in the data flow, not by build order (for build order and per-phase
oracles, see §10 of `grib-to-inference-pipeline.md`).

1. **GRIB decode.** Nothing today. Runtime scope is GRIB2 + `grid_ccsds` + bitmaps + the
   `bitsPerValue = 0` case; GRIB1 + `grid_simple` can be kept off the runtime path entirely by
   decoding `lsm.grib` once offline (§2.3, §4.3 — it is `constant_in_time`). The `.index`
   sidecars of §2.4 make message selection a seek rather than a scan.
2. **Regrid A → B.** 1,038,240 → 542,080 per field, as a cached sparse matrix and a CSR spmv
   (§3.3).
3. **Variable routing.** `(paramId, typeOfLevel, level)` → variable name, covering the six
   aliased cases of §4.5. `metadata.var_to_input_channel` already provides the second half of
   the map (name → channel); the first half does not exist.
4. **The two field-level transforms.** `mwd` → `cos_mwd`/`sin_mwd`, and `apply-mask` on
   `sd`/`swvl1`/`swvl2` against `lsm == 0`. Both operate on named fields before assembly.
5. **The 9 computed forcings.** `metadata.computed_forcing` lists them; nothing computes them.
6. **ConstantImputer.** `metadata.imputer_zero` is parsed at `src/metadata.rs:168` and never
   read.
7. **InputNormalizer.** A pure affine map — the four tensors `_norm_mul`, `_norm_add`,
   `_input_idx`, `_output_idx` are in the checkpoint but currently land in `unused` at load
   (`src/main.rs:103`). No method dispatch is needed; the normalisation method was resolved
   into coefficients at training time.
8. **Assembly from real fields.** Replace the `Tensor::zeros` at `src/main.rs:211-219`, and
   port the `check[]` invariant of `tensors.py:168-172` — a `[bool; 106]` asserted before every
   forward. A missing forcing otherwise produces a plausible-looking, quietly wrong forecast
   that will not show up in a one-step test.

Then, to close the loop but not required for one populated input tensor:

9. **Boundings.** Explicit `TODO(saiputravu)` at `src/aifs.rs:188-190`. Output-side, applied in
   list order after the residual; `FractionBounding` reads variables the earlier entries have
   already clamped, so the order is load-bearing.
10. **A second retrieval at t − 6h.** Everything on disk is `stepRange = 0` (§4.5).
11. **`sd`.** Absent from both sample files (§4.5). A catalogue question, not a code question.

Items 1–8 are the minimum to hold the entire model input in memory. Items 1 and 2 are the only
ones that are genuinely hard; 3–8 are each a few hundred lines against a schema that
`src/metadata.rs` already exposes in full.

---

## Appendix — reproducing every number

All commands are runnable from
`/Users/sai/Documents/projects/airglow/aifsv2/`. The venv at
`data/aifs-single-mse-2.0/quiet_grub/.venv/bin/python` has pygrib, earthkit-data 0.20.0,
earthkit-regrid 0.5.1, eccodes 2.47.0 and anemoi-transform 0.4.3.

```bash
# §1.1  GRIB2 section lengths sum to totalLength; no eccodes needed
python3 -c "
import json
m = json.load(open('data/20260810000000-0h-oper-fc_keys.json'))['messages'][0]
print(m['shortName'], m['level'], m['typeOfLevel'])
print([m[f'section{i}Length'] for i in range(9)])
print(sum(m[f'section{i}Length'] for i in range(9)), m['totalLength'])
"                                        # -> w 600 isobaricInhPa
                                         #    [16,21,17,72,34,25,6,1260051,4]
                                         #    1260246 1260246

# §1.1  GRIB1 section lengths, and §2.3 grid keys
python3 -c "
import json
m = json.load(open('data/lsm_keys.json'))['messages'][0]
for k in ('editionNumber','gridType','N','packingType','bitsPerValue',
          'referenceValue','binaryScaleFactor','decimalScaleFactor',
          'numberOfDataPoints','section0Length','section1Length',
          'section2Length','section4Length','totalLength','bitmapPresent'):
    print(f'{k:24s}', m[k])
pl = m['pl']
print('pl', len(pl), sum(pl), pl[:8], max(pl), pl == pl[::-1])
"                                        # -> 1 reduced_gg 320 grid_simple 24 ...
                                         #    pl 640 542080 [18,25,36,40,45,50,60,64] 1280 True

# §2.1  key tallies over all 126 messages
python3 -c "
import json, collections
m = json.load(open('data/20260810000000-0h-oper-fc_keys.json'))['messages']
for k in ('editionNumber','gridType','packingType','bitsPerValue','numberOfDataPoints',
          'Ni','Nj','bitmapPresent','scanningMode','stepRange',
          'latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees',
          'iDirectionIncrementInDegrees'):
    print(f'{k:36s}', dict(collections.Counter(str(x[k]) for x in m)))
"

# §2.1  per-parameter inventory
python3 -c "
import json, collections
m = json.load(open('data/20260810000000-0h-oper-fc_keys.json'))['messages']
by = collections.defaultdict(list)
for x in m: by[(x['shortName'], x['paramId'], x['typeOfLevel'])].append(x['level'])
for (sn,pid,tl), lv in sorted(by.items()):
    print(f'{sn:8s} {pid:8d} {tl:18s} n={len(lv):3d} {sorted(set(lv))}')
"

# §2.2  the wave file
data/aifs-single-mse-2.0/quiet_grub/.venv/bin/python -c "
import pygrib
for m in pygrib.open('data/20260810000000-0h-wave-fc.grib2'):
    print(f'#{m.messagenumber:<3d} {m.shortName:8s} {m.paramId:<8d} {m.typeOfLevel:16s}'
          f' lvl={m.level} {m.gridType} {m.numberOfDataPoints} pack={m.packingType}'
          f' bpv={m.bitsPerValue} bmp={m[\"bitmapPresent\"]}')
"

# §2.4  the .index sidecars tile the files exactly
python3 -c "
import json, os
for f in ('data/20260810000000-0h-oper-fc', 'data/20260810000000-0h-wave-fc'):
    r = [json.loads(l) for l in open(f + '.index') if l.strip()]
    size = os.path.getsize(f + '.grib2')
    ok = all(r[i]['_offset'] + r[i]['_length'] == r[i+1]['_offset'] for i in range(len(r)-1))
    print(f, len(r), size, sum(x['_length'] for x in r), 'contiguous', ok)
"                                        # -> 126 91085441 91085441 contiguous True
                                         #    11   8694454  8694454 contiguous True

# §3.2  the N320 coordinates ship as raw f64, no .npy header
python3 -c "
import numpy as np
D = 'data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata'
lat = np.fromfile(f'{D}/latitudes.numpy',  dtype='<f8')
lon = np.fromfile(f'{D}/longitudes.numpy', dtype='<f8')
print(lat.shape, len(np.unique(lat)), lat.min(), lat.max())   # (542080,) 640 ±89.78487690721863
print(lon[:3], lon.min(), lon.max())                          # [0. 20. 40.] 0.0 359.71875
rows = np.array(sorted(set(lat.tolist()), reverse=True))
d = -np.diff(rows)
print(d.min(), d.max())                                       # 0.278674 0.281030
"

# §3.2  the pl plateau: 1280 held flat across 148 rows, |lat| <= 20.66
python3 -c "
import json, numpy as np
pl  = json.load(open('data/lsm_keys.json'))['messages'][0]['pl']
lat = np.fromfile('data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/latitudes.numpy', dtype='<f8')
rows = sorted(set(lat.tolist()), reverse=True)
idx = [i for i, p in enumerate(pl) if p == max(pl)]
print(max(pl), idx[0], idx[-1], len(idx), rows[idx[0]], rows[idx[-1]])
print(640 * 1280 - sum(pl))
"                                        # -> 1280 246 393 148 20.6557 -20.6557
                                         #    277120

# §4.2, §4.3  roles and provenance
data/aifs-single-mse-2.0/quiet_grub/.venv/bin/python -c "
import json
md = json.load(open('data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/ai-models.json'))
vs, vm = md['dataset']['variables'], md['dataset']['variables_metadata']
di = md['data_indices']
for space in ('data','model'):
    for io in ('input','output'):
        print(space, io, {k: len(v) for k, v in di[space][io].items()})
comp = [v for v in vs if vm.get(v, {}).get('computed_forcing')]
const = [v for v in vs if vm.get(v, {}).get('constant_in_time')]
forc = [vs[i] for i in di['data']['input']['forcing']]
print('computed', len(comp), comp)
print('constant', len(const), const)
print('retrieved forcings', sorted(set(forc) - set(comp)))
"                                        # -> 106/120 ... 9 / 9 / [lsm, sdor, slor, wmb, z]

# §4.5  the 97-variable join against all three files
data/aifs-single-mse-2.0/quiet_grub/.venv/bin/python -c "
import json, pygrib, collections
md = json.load(open('data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata/ai-models.json'))
vs, vm = md['dataset']['variables'], md['dataset']['variables_metadata']
inp = [vs[i] for i in md['data_indices']['data']['input']['full']]
comp = {v for v in vs if vm.get(v, {}).get('computed_forcing')}
need = [v for v in inp if v not in comp]
avail = collections.defaultdict(list)
for p in ('data/20260810000000-0h-oper-fc.grib2',
          'data/20260810000000-0h-wave-fc.grib2', 'data/lsm.grib'):
    for m in pygrib.open(p):
        avail[(m.shortName, m.typeOfLevel)].append(m.level)
def hit(v):
    if '_' in v and v.rsplit('_',1)[1].isdigit():
        b, l = v.rsplit('_',1)
        return int(l) in avail.get((b,'isobaricInhPa'), [])
    return any(sn == v for sn, _ in avail)
print('need', len(need), 'matched', sum(map(hit, need)),
      'unmatched', [v for v in need if not hit(v)])
"                                        # -> need 97 matched 90 unmatched
                                         #    ['cos_mwd','sd','sin_mwd','stl1','stl2','swvl1','swvl2']

# §3.3  the regrid is a sparse matmul, not an algorithm
sed -n '30,41p' data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/regrid/interpolate.py
grep -n '_SYSTEM_URL' data/aifs-single-mse-2.0/quiet_grub/.venv/lib/python3.12/site-packages/earthkit/regrid/db.py

# §5.1  lagged = [-6h, 0h]
sed -n '227,234p' /Users/sai/Documents/projects/anemoi/anemoi-inference/src/anemoi/inference/metadata.py
```
