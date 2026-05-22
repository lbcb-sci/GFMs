#!/usr/bin/env bash
# =============================================================================
# Download cDNA (*.cdna.all.fa.gz) for every species from Ensembl Release 115
#
# Usage:
#   chmod +x download_ensembl_cdna.sh
#   ./download_ensembl_cdna.sh [OUTPUT_DIR]
#
# Default output directory: ./ensembl_cdna_r115
# Requires: curl (or wget), lftp (for reliable FTP directory listing)
#           If lftp is not installed, falls back to curl.
# =============================================================================

set -euo pipefail

RELEASE=115
BASE_URL="https://ftp.ensembl.org/pub/release-${RELEASE}/fasta"
OUTDIR="${1:-./ensembl_cdna_r115}"
MAX_PARALLEL=4
LOG_FILE="${OUTDIR}/download.log"

mkdir -p "$OUTDIR"

echo "============================================="
echo " Ensembl Release ${RELEASE} - cDNA Downloader"
echo "============================================="
echo "Output directory: $OUTDIR"
echo ""

# --- Step 1: Get list of species directories --------------------------------
echo "[1/3] Fetching species list..."

SPECIES_LIST="${OUTDIR}/.species_list.txt"

# Try lftp first (most reliable for FTP listing), fall back to curl
if command -v lftp &>/dev/null; then
    lftp -e "cls --sort=name ftp://ftp.ensembl.org/pub/release-${RELEASE}/fasta/; quit" 2>/dev/null \
        | sed 's|/$||' | grep -v '^\.' > "$SPECIES_LIST"
else
    # Fallback: curl FTP directory listing
    curl -s "ftp://ftp.ensembl.org/pub/release-${RELEASE}/fasta/" \
        | awk '{print $NF}' | sed 's|/$||' | grep -v '^\.' > "$SPECIES_LIST"
fi

TOTAL=$(wc -l < "$SPECIES_LIST")
echo "   Found $TOTAL species."
echo ""

# --- Step 2: Download cdna.all.fa.gz for each species -----------------------
echo "[2/3] Downloading cdna.all.fa.gz files (${MAX_PARALLEL} parallel)..."
echo ""

download_cdna() {
    local species="$1"
    local outdir="$2"
    local base_url="$3"
    local release="$4"
    local dest_dir="${outdir}/${species}"
    local cdna_url="${base_url}/${species}/cdna/"

    mkdir -p "$dest_dir"

    # List files in the cdna directory and find the *.cdna.all.fa.gz file
    local filename
    filename=$(curl -s "ftp://ftp.ensembl.org/pub/release-${release}/fasta/${species}/cdna/" \
        | awk '{print $NF}' | grep '\.cdna\.all\.fa\.gz$' | head -1 || true)

    if [[ -z "$filename" ]]; then
        echo "  [SKIP] ${species}: no cdna.all.fa.gz found" | tee -a "$outdir/download.log"
        return 0
    fi

    local dest_file="${dest_dir}/${filename}"
    local download_url="https://ftp.ensembl.org/pub/release-${release}/fasta/${species}/cdna/${filename}"

    # Skip if already downloaded and non-empty
    if [[ -s "$dest_file" ]]; then
        echo "  [EXISTS] ${species}: ${filename}" | tee -a "$outdir/download.log"
        return 0
    fi

    if curl -sS -f -o "$dest_file" --retry 3 --retry-delay 5 "$download_url" 2>>"$outdir/download.log"; then
        local size
        size=$(du -h "$dest_file" | cut -f1)
        echo "  [OK]   ${species}: ${filename} (${size})" | tee -a "$outdir/download.log"
    else
        echo "  [FAIL] ${species}: ${filename}" | tee -a "$outdir/download.log"
        rm -f "$dest_file"
    fi
}

export -f download_cdna

# Run downloads in parallel using xargs
cat "$SPECIES_LIST" \
    | xargs -P "$MAX_PARALLEL" -I {} bash -c 'download_cdna "$@"' _ {} "$OUTDIR" "$BASE_URL" "$RELEASE"

# --- Step 3: Summary --------------------------------------------------------
echo ""
echo "[3/3] Summary"
echo "---------------------------------------------"

DOWNLOADED=$(find "$OUTDIR" -name "*.cdna.all.fa.gz" -size +0 | wc -l)
FAILED=$(grep -c "\[FAIL\]" "$LOG_FILE" 2>/dev/null || echo 0)
SKIPPED=$(grep -c "\[SKIP\]" "$LOG_FILE" 2>/dev/null || echo 0)

echo "  Total species:  $TOTAL"
echo "  Downloaded:     $DOWNLOADED"
echo "  Already existed: $(grep -c "\[EXISTS\]" "$LOG_FILE" 2>/dev/null || echo 0)"
echo "  Skipped (no file): $SKIPPED"
echo "  Failed:         $FAILED"
echo ""
echo "Files saved to: $OUTDIR/<species>/*.cdna.all.fa.gz"
echo "Log file: $LOG_FILE"
echo ""
echo "Done!"
