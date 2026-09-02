#!/usr/bin/env bash
# Rebuild ./data from nothing: download the HuggingFace artifacts, unzip the
# checkpoint, convert everything the Rust runtime reads, and fetch the GRIB
# input pair. Idempotent — every step is skipped when its output already exists,
# so a partial run can simply be re-run.
#
# Needs: curl, python3, uv (the conversion scripts run via `uv run`, which
# syncs the pyproject environment on first use — Artifactory credentials
# required for that first sync; see the header of pyproject.toml).
#
# The GRIB base time defaults to the pair hardcoded in src/main.rs. ECMWF open
# data only retains ~4 days, so for a fresh setup pass a newer time and update
# the OPER/WAVE_PATH constants to match:
#   BASE_TIME=2026-09-02T00 scripts/generate-data.sh
set -euo pipefail

cd "$(dirname "$0")/.."

HF_BASE=https://huggingface.co/ecmwf/aifs-single-2.0/resolve/main
CKPT=data/aifs-single-mse-2.0.ckpt

# Timestep 1 of the model input; timestep 0 is derived 6 h earlier (multistep
# spacing). Both oper and wave streams are fetched for each.
BASE_TIME=${BASE_TIME:-2026-08-31T00}
PREV_TIME=$(date -u -d "${BASE_TIME/T/ }:00 UTC 6 hours ago" +%Y-%m-%dT%H)

mkdir -p data/grib

fetch() { # fetch <url> <dest> — atomic: no partial file left on interrupt
    local url=$1 dest=$2
    if [ -e "$dest" ]; then
        echo "skip (exists): $dest"
        return
    fi
    echo "downloading $dest"
    curl -fL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

fetch "$HF_BASE/aifs-single-mse-2.0.ckpt" "$CKPT"
fetch "$HF_BASE/inference.yaml" data/inference.yaml
fetch "$HF_BASE/lsm.grib" data/grib/lsm.grib

# The ckpt is a torch zip whose internal root is quiet_grub/ (anemoi's codename
# for this model). main.rs reads data/quiet_grub/anemoi-metadata directly, so it
# is unzipped in place rather than into a temp dir.
if [ ! -d data/quiet_grub ]; then
    echo "unzipping $CKPT"
    python3 -m zipfile -e "$CKPT" data/
fi

# Weights + processor arrays; also writes <stem>_metadata.json, which both the
# Rust runtime and download_opendata.py read. Guard on the json: it is written
# last, so its presence implies the safetensors is complete.
if [ ! -e data/aifs-single-mse-2.0_metadata.json ]; then
    uv run python scripts/ckpt_to_safetensors.py "$CKPT"
fi

# Graph connectivity (edge_index/lengths/dirs), which the state_dict lacks.
if [ ! -e data/aifs-single-mse-2.0_graph.safetensors ]; then
    uv run python scripts/extract_graph.py "$CKPT"
fi

# The 0.25 degree -> N320 interpolation operator, from ECMWF's matrix repository.
if [ ! -e data/regrid-0p25-to-n320.safetensors ]; then
    uv run python scripts/fetch_regrid_matrix.py
fi

# Output GRIB templates, decoded from anemoi-inference's builtin index. Committed, so
# this only runs on a checkout that has lost them; needs the anemoi-inference repo.
if [ ! -e data/templates/n320-pl.grib2 ] || [ ! -e data/templates/n320-sfc.grib2 ]; then
    uv run scripts/extract_grib_templates.py --anemoi "${ANEMOI_INFERENCE:-../anemoi-inference}"
fi

# The two input base times, 6 h apart; existing files are skipped by the script.
uv run python scripts/download_opendata.py \
    --start "$PREV_TIME" --end "$BASE_TIME" --freq 6 \
    --output-dir data/grib \
    --metadata data/aifs-single-mse-2.0_metadata.json

echo
echo "data/ ready. If BASE_TIME was overridden, update OPER_PATH/WAVE_PATH(_PREV)"
echo "in src/main.rs to ${PREV_TIME} / ${BASE_TIME}."
