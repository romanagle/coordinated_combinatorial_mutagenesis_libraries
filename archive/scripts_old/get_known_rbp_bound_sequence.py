#!/usr/bin/env python3
"""Fetch an endogenous RNA sequence bound by an RBP.

Primary mode queries ENCODE eCLIP peak files by RBP gene symbol, selects a
released BED peak, then uses the UCSC API to retrieve the genomic sequence.
The output sequence is reported as RNA (T converted to U).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


ENCODE = "https://www.encodeproject.org"
UCSC_SEQUENCE = "https://api.genome.ucsc.edu/getData/sequence"

RNCMPT_ALIASES = {
    # Local project aliases seen in scripts/notebooks. A complete RNCMPT->gene
    # map should come from CISBP-RNA metadata.
    "RNCMPT00176": "MSI1",
    "RNCMPT00186": "PCBP1",
}


@dataclass(frozen=True)
class Peak:
    chrom: str
    start: int
    end: int
    name: str
    score: float
    strand: str


def get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "rbp-bound-sequence/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for JSON URL: {url}") from exc


def open_text_url(url: str):
    req = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": "rbp-bound-sequence/1.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for text URL: {url}") from exc
    if url.endswith(".gz"):
        return gzip.open(resp, mode="rt")
    return resp


def encode_search_url(params: list[tuple[str, str]]) -> str:
    return f"{ENCODE}/search/?{urllib.parse.urlencode(params)}"


def search_encode_files(rbp: str, assembly: str, verbose: bool = False) -> list[dict]:
    base_params = [
        ("type", "File"),
        ("assay_title", "eCLIP"),
        ("file_format", "bed"),
        ("assembly", assembly),
        ("status", "released"),
        ("format", "json"),
        ("limit", "all"),
    ]
    urls = [
        encode_search_url(base_params + [("target.label", rbp)]),
        encode_search_url(base_params + [("searchTerm", rbp)]),
    ]

    seen = set()
    files: list[dict] = []
    for url in urls:
        if verbose:
            print(f"[query] {url}", file=sys.stderr)
        try:
            data = get_json(url)
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                if verbose:
                    print(f"[query skipped] no ENCODE hits for: {url}", file=sys.stderr)
                continue
            raise
        for item in data.get("@graph", []):
            accession = item.get("accession")
            if accession and accession not in seen:
                seen.add(accession)
                files.append(item)
    return files


def file_rank(file_obj: dict) -> tuple[int, int, str]:
    output_type = str(file_obj.get("output_type", "")).lower()
    preferred = 0
    if "idr" in output_type:
        preferred -= 20
    if "peak" in output_type:
        preferred -= 10
    if "replicated" in output_type:
        preferred -= 5
    if file_obj.get("preferred_default"):
        preferred -= 5
    date = str(file_obj.get("date_created", ""))
    return (preferred, 0 if file_obj.get("href") else 1, date)


def choose_encode_bed(files: list[dict]) -> dict:
    peak_files = [
        f
        for f in files
        if "peak" in str(f.get("output_type", "")).lower()
        and f.get("href")
        and str(f.get("file_format", "")).lower() == "bed"
    ]
    if not peak_files:
        raise RuntimeError("No released ENCODE eCLIP BED peak files found.")
    return sorted(peak_files, key=file_rank)[0]


def iter_bed_peaks(lines: Iterable[str], max_len: int) -> Iterable[Peak]:
    for line in lines:
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        chrom, start_s, end_s = fields[:3]
        try:
            start, end = int(start_s), int(end_s)
        except ValueError:
            continue
        if end <= start or end - start > max_len:
            continue
        name = fields[3] if len(fields) > 3 else "."
        score = parse_score(fields)
        strand = fields[5] if len(fields) > 5 and fields[5] in {"+", "-"} else "+"
        yield Peak(chrom, start, end, name, score, strand)


def parse_score(fields: list[str]) -> float:
    for idx in (6, 7, 4):
        if len(fields) > idx:
            try:
                return float(fields[idx])
            except ValueError:
                pass
    return 0.0


def choose_peak(bed_url: str, max_len: int, rank: int) -> Peak:
    with open_text_url(bed_url) as lines:
        peaks = sorted(iter_bed_peaks(lines, max_len), key=lambda p: p.score, reverse=True)
    if not peaks:
        raise RuntimeError(f"No BED peaks shorter than or equal to {max_len} nt found.")
    if rank < 1 or rank > len(peaks):
        raise RuntimeError(f"--rank must be between 1 and {len(peaks)} for this BED file.")
    return peaks[rank - 1]


def fetch_sequence(assembly: str, peak: Peak) -> str:
    params = {
        "genome": assembly,
        "chrom": peak.chrom,
        "start": str(peak.start),
        "end": str(peak.end),
    }
    if peak.strand == "-":
        params["revComp"] = "1"
    query = ";".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items())
    data = get_json(f"{UCSC_SEQUENCE}?{query}")
    dna = data.get("dna")
    if not dna:
        raise RuntimeError(f"UCSC did not return sequence for {peak.chrom}:{peak.start}-{peak.end}.")
    return dna.upper().replace("T", "U")


def resolve_rbp(rbp: str) -> str:
    key = rbp.upper()
    return RNCMPT_ALIASES.get(key, rbp.upper())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Input an RBP gene symbol and output one known endogenous bound RNA sequence."
    )
    parser.add_argument("rbp", help="RBP gene symbol, e.g. MSI1, PCBP1, RBFOX2. Some RNCMPT aliases work.")
    parser.add_argument("--assembly", default="GRCh38", help="ENCODE assembly, usually GRCh38 or mm10.")
    parser.add_argument("--ucsc-genome", default="hg38", help="UCSC genome name used for sequence extraction.")
    parser.add_argument("--max-peak-len", type=int, default=300, help="Skip broader peaks longer than this.")
    parser.add_argument("--rank", type=int, default=1, help="Return the Nth highest-scoring usable peak.")
    parser.add_argument("--bed-url", help="Use a BED/BED.gz URL directly instead of searching ENCODE.")
    parser.add_argument("--verbose", action="store_true", help="Print queried URLs and selected source.")
    args = parser.parse_args()

    rbp = resolve_rbp(args.rbp)
    file_obj = None
    if args.bed_url:
        bed_url = args.bed_url
    else:
        files = search_encode_files(rbp, args.assembly, verbose=args.verbose)
        file_obj = choose_encode_bed(files)
        bed_url = urllib.parse.urljoin(ENCODE, file_obj["href"])
        if args.verbose:
            print(
                f"[selected] {file_obj.get('accession')} {file_obj.get('output_type')} {bed_url}",
                file=sys.stderr,
            )

    peak = choose_peak(bed_url, args.max_peak_len, args.rank)
    if args.verbose:
        print(f"[peak] {peak}", file=sys.stderr)
    sequence = fetch_sequence(args.ucsc_genome, peak)

    source = file_obj.get("accession") if file_obj else bed_url
    output_type = file_obj.get("output_type", "BED peaks") if file_obj else "BED peaks"
    print(f">{rbp}|{peak.chrom}:{peak.start}-{peak.end}({peak.strand})|score={peak.score}|source={source}|{output_type}")
    print(sequence)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
