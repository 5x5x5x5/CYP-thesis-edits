#!/usr/bin/env python3
"""Fetch PubChem SMILES for all unique SIDs across CYP splits and precompute RDKit features.

Outputs:
  - notebooks/data/sid_smiles.csv: SID,CID,SMILES,CanonicalSMILES,ScaffoldSMILES
  - notebooks/data/rdkit_features.npz: compressed numpy archive indexed by sorted `sids`
      sids (int64), valid_mask (bool),
      ecfp4_bits (uint8, 2048), ecfp4_counts (uint8, 2048, clipped to 255),
      fcfp4_bits (uint8, 1024), maccs (uint8, 167), rdkit_fp (uint8, 2048),
      rdkit_desc (float32, N_desc), rdkit_desc_names (str, N_desc)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ISOZYMES = ("1a2", "2c9", "2c19", "2d6", "3a4")
DATA_DIR = Path(__file__).resolve().parent / "data"
AID_CONCISE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1851/concise/CSV"
SID_TO_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/cids/JSON"
CID_PROP_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/"
    "IsomericSMILES,CanonicalSMILES/CSV"
)


def extract_unique_sids(data_dir: Path) -> tuple[list[int], dict[str, set[int]]]:
    """Return sorted unique SIDs and per-split SID sets."""
    split_sids: dict[str, set[int]] = {}
    all_sids: set[int] = set()
    for iso in ISOZYMES:
        for prefix in ("training", "test"):
            split_name = f"{prefix}{iso}"
            path = data_dir / f"{split_name}.csv"
            df = pd.read_csv(path, usecols=["SID"])
            sids = set(df["SID"].astype(int).tolist())
            split_sids[split_name] = sids
            all_sids.update(sids)
    return sorted(all_sids), split_sids


def http_request_with_backoff(
    url: str,
    data: bytes | None = None,
    timeout: int = 60,
    max_retries: int = 6,
    base_delay: float = 1.0,
) -> bytes:
    """Perform GET/POST with exponential backoff on transient HTTP/network errors."""
    headers = {"User-Agent": "CYP-thesis-edits/1.0 (academic research)"}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(
                    f"  [HTTP {exc.code}] retry {attempt + 1}/{max_retries - 1} in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(
                    f"  [Network error: {exc}] retry {attempt + 1}/{max_retries - 1} in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"Exhausted retries for {url}")


def fetch_aid1851_sid_to_cid(target_sids: set[int]) -> dict[int, int]:
    """Download AID 1851 concise CSV and extract SID -> CID for target SIDs."""
    print(f"Downloading AID 1851 concise CSV from {AID_CONCISE_URL}...", flush=True)
    t0 = time.time()
    raw = http_request_with_backoff(AID_CONCISE_URL, timeout=120)
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    sid_to_cid: dict[int, int] = {}
    for row in reader:
        sid_str = (row.get("SID") or "").strip()
        cid_str = (row.get("CID") or "").strip()
        if not sid_str or not cid_str:
            continue
        try:
            sid = int(sid_str)
            if sid in target_sids and sid not in sid_to_cid:
                sid_to_cid[sid] = int(cid_str)
        except ValueError:
            continue
    print(
        f"  AID 1851 concise CSV parsed in {time.time() - t0:.1f}s: "
        f"matched {len(sid_to_cid):,}/{len(target_sids):,} SIDs.",
        flush=True,
    )
    return sid_to_cid


def fetch_fallback_sid_to_cid(missing_sids: list[int], batch_size: int = 400) -> dict[int, int]:
    """Fallback batch POST to substance/sid/cids/JSON for SIDs missing from AID 1851."""
    if not missing_sids:
        return {}
    print(f"Fetching fallback SID->CID for {len(missing_sids):,} SIDs...", flush=True)
    result: dict[int, int] = {}
    for start in range(0, len(missing_sids), batch_size):
        batch = missing_sids[start : start + batch_size]
        payload = urllib.parse.urlencode({"sid": ",".join(map(str, batch))}).encode("utf-8")
        try:
            raw = http_request_with_backoff(SID_TO_CID_URL, data=payload, timeout=60)
            obj = json.loads(raw.decode("utf-8"))
            for info in obj.get("InformationList", {}).get("Information", []):
                sid = info.get("SID")
                cids = info.get("CID")
                if sid is not None and cids:
                    result[int(sid)] = int(cids[0])
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                pass
            elif len(batch) > 1:
                mid = len(batch) // 2
                result.update(fetch_fallback_sid_to_cid(batch[:mid], batch_size=max(1, mid)))
                result.update(fetch_fallback_sid_to_cid(batch[mid:], batch_size=max(1, len(batch) - mid)))
            else:
                print(f"  Skipping SID {batch[0]}: HTTP {exc.code}", flush=True)
        time.sleep(0.25)
    print(f"  Fallback resolved {len(result):,}/{len(missing_sids):,} SIDs.", flush=True)
    return result


def fetch_cid_smiles(cids: list[int], batch_size: int = 500) -> dict[int, tuple[str, str]]:
    """Batch-fetch (SMILES, ConnectivitySMILES) for sorted unique CIDs via PUG-REST POST."""
    cid_to_smiles: dict[int, tuple[str, str]] = {}
    total = len(cids)
    n_batches = math.ceil(total / batch_size)
    print(f"Fetching SMILES for {total:,} unique CIDs in {n_batches} batches...", flush=True)
    t0 = time.time()

    def _fetch_chunk(chunk: list[int]) -> None:
        payload = urllib.parse.urlencode({"cid": ",".join(map(str, chunk))}).encode("utf-8")
        try:
            raw = http_request_with_backoff(CID_PROP_URL, data=payload, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404) and len(chunk) > 1:
                mid = len(chunk) // 2
                _fetch_chunk(chunk[:mid])
                time.sleep(0.25)
                _fetch_chunk(chunk[mid:])
                return
            if exc.code in (400, 404):
                print(f"  CID {chunk[0]} not found (HTTP {exc.code})", flush=True)
                return
            raise

        text = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            cid_str = (row.get("CID") or "").strip()
            if not cid_str:
                continue
            cid = int(cid_str)
            iso_smi = (
                row.get("SMILES")
                or row.get("IsomericSMILES")
                or row.get("ConnectivitySMILES")
                or row.get("CanonicalSMILES")
                or ""
            ).strip()
            conn_smi = (
                row.get("ConnectivitySMILES")
                or row.get("CanonicalSMILES")
                or iso_smi
            ).strip()
            cid_to_smiles[cid] = (iso_smi, conn_smi)

    for idx, start in enumerate(range(0, total, batch_size), start=1):
        chunk = cids[start : start + batch_size]
        _fetch_chunk(chunk)
        if idx % 10 == 0 or idx == n_batches:
            print(
                f"  Batch {idx}/{n_batches}: {len(cid_to_smiles):,}/{total:,} CIDs "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
        time.sleep(0.25)

    return cid_to_smiles


# Worker globals for multiprocessing
_LFC = None
_ECFP4_GEN = None
_FCFP4_GEN = None
_RDKIT_GEN = None
_DESC_FUNCS = None


def _init_worker() -> None:
    global _LFC, _ECFP4_GEN, _FCFP4_GEN, _RDKIT_GEN, _DESC_FUNCS
    RDLogger.DisableLog("rdApp.*")
    _LFC = rdMolStandardize.LargestFragmentChooser()
    _ECFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _FCFP4_GEN = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    _RDKIT_GEN = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
    _DESC_FUNCS = [fn for _, fn in Descriptors._descList]


def _featurize_chunk(
    items: list[tuple[int, str]],
) -> tuple[
    list[int],
    list[bool],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = len(items)
    n_desc = len(_DESC_FUNCS)

    sids_out: list[int] = []
    valid_out: list[bool] = []
    canon_out: list[str] = []
    scaff_out: list[str] = []

    ecfp4_bits = np.zeros((n, 2048), dtype=np.uint8)
    ecfp4_counts = np.zeros((n, 2048), dtype=np.uint8)
    fcfp4_bits = np.zeros((n, 1024), dtype=np.uint8)
    maccs = np.zeros((n, 167), dtype=np.uint8)
    rdkit_fp = np.zeros((n, 2048), dtype=np.uint8)
    rdkit_desc = np.full((n, n_desc), np.nan, dtype=np.float32)

    f32_max = float(np.finfo(np.float32).max)

    for i, (sid, smi) in enumerate(items):
        sids_out.append(sid)
        if not smi:
            valid_out.append(False)
            canon_out.append("")
            scaff_out.append("")
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_out.append(False)
            canon_out.append("")
            scaff_out.append("")
            continue

        try:
            mol = _LFC.choose(mol)
            Chem.SanitizeMol(mol)
        except Exception:
            valid_out.append(False)
            canon_out.append("")
            scaff_out.append("")
            continue

        canon_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        try:
            scaff_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaff_smi = ""

        valid_out.append(True)
        canon_out.append(canon_smi)
        scaff_out.append(scaff_smi)

        ecfp4_bits[i] = _ECFP4_GEN.GetFingerprintAsNumPy(mol)
        counts = _ECFP4_GEN.GetCountFingerprintAsNumPy(mol)
        ecfp4_counts[i] = np.minimum(counts, 255).astype(np.uint8)
        fcfp4_bits[i] = _FCFP4_GEN.GetFingerprintAsNumPy(mol)
        maccs[i] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)
        rdkit_fp[i] = _RDKIT_GEN.GetFingerprintAsNumPy(mol)

        for j, fn in enumerate(_DESC_FUNCS):
            try:
                val = float(fn(mol))
                if math.isfinite(val) and abs(val) <= f32_max:
                    rdkit_desc[i, j] = val
            except Exception:
                pass

    return (
        sids_out,
        valid_out,
        canon_out,
        scaff_out,
        ecfp4_bits,
        ecfp4_counts,
        fcfp4_bits,
        maccs,
        rdkit_fp,
        rdkit_desc,
    )


def compute_rdkit_features(
    sids: list[int],
    sid_to_raw_smiles: dict[int, str],
    n_workers: int,
) -> tuple[dict[int, tuple[str, str, bool]], dict[str, np.ndarray]]:
    """Compute salt-stripped canonical SMILES, scaffolds, fingerprints, and 2D descriptors."""
    n = len(sids)
    desc_names = np.array([name for name, _ in Descriptors._descList], dtype=str)
    items = [(sid, sid_to_raw_smiles.get(sid, "")) for sid in sids]

    chunk_size = max(1, math.ceil(n / (n_workers * 4)))
    chunks = [items[i : i + chunk_size] for i in range(0, n, chunk_size)]

    print(
        f"Computing RDKit features for {n:,} SIDs across {n_workers} workers "
        f"({len(chunks)} chunks, {len(desc_names)} descriptors)...",
        flush=True,
    )
    t0 = time.time()

    all_sids: list[int] = []
    all_valid: list[bool] = []
    all_canon: list[str] = []
    all_scaff: list[str] = []
    ecfp4_bits_list: list[np.ndarray] = []
    ecfp4_counts_list: list[np.ndarray] = []
    fcfp4_bits_list: list[np.ndarray] = []
    maccs_list: list[np.ndarray] = []
    rdkit_fp_list: list[np.ndarray] = []
    rdkit_desc_list: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        for res in pool.map(_featurize_chunk, chunks):
            (
                s_out,
                v_out,
                c_out,
                sc_out,
                e_bits,
                e_cnts,
                f_bits,
                m_keys,
                r_fp,
                r_desc,
            ) = res
            all_sids.extend(s_out)
            all_valid.extend(v_out)
            all_canon.extend(c_out)
            all_scaff.extend(sc_out)
            ecfp4_bits_list.append(e_bits)
            ecfp4_counts_list.append(e_cnts)
            fcfp4_bits_list.append(f_bits)
            maccs_list.append(m_keys)
            rdkit_fp_list.append(r_fp)
            rdkit_desc_list.append(r_desc)

    print(f"  RDKit featurization finished in {time.time() - t0:.1f}s.", flush=True)

    meta: dict[int, tuple[str, str, bool]] = {
        sid: (canon, scaff, valid)
        for sid, canon, scaff, valid in zip(all_sids, all_canon, all_scaff, all_valid)
    }

    arrays = {
        "sids": np.array(all_sids, dtype=np.int64),
        "valid_mask": np.array(all_valid, dtype=bool),
        "ecfp4_bits": np.vstack(ecfp4_bits_list),
        "ecfp4_counts": np.vstack(ecfp4_counts_list),
        "fcfp4_bits": np.vstack(fcfp4_bits_list),
        "maccs": np.vstack(maccs_list),
        "rdkit_fp": np.vstack(rdkit_fp_list),
        "rdkit_desc": np.vstack(rdkit_desc_list),
        "rdkit_desc_names": desc_names,
    }
    return meta, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument(
        "--skip-pubchem",
        action="store_true",
        help="Reuse existing sid_smiles.csv (SID,CID,SMILES) instead of querying PubChem.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    csv_out = data_dir / "sid_smiles.csv"
    npz_out = data_dir / "rdkit_features.npz"

    sids, split_sids = extract_unique_sids(data_dir)
    print(f"Extracted {len(sids):,} unique SIDs across {len(split_sids)} splits.", flush=True)

    sid_to_cid: dict[int, int] = {}
    sid_to_raw_smiles: dict[int, str] = {}

    if args.skip_pubchem and csv_out.exists():
        print(f"Loading existing PubChem mapping from {csv_out}...", flush=True)
        existing = pd.read_csv(csv_out)
        for row in existing.itertuples(index=False):
            sid = int(row.SID)
            if pd.notna(row.CID):
                sid_to_cid[sid] = int(row.CID)
            if isinstance(row.SMILES, str) and row.SMILES:
                sid_to_raw_smiles[sid] = row.SMILES
    else:
        sid_set = set(sids)
        sid_to_cid = fetch_aid1851_sid_to_cid(sid_set)
        missing_sids = [s for s in sids if s not in sid_to_cid]
        if missing_sids:
            fallback = fetch_fallback_sid_to_cid(missing_sids)
            sid_to_cid.update(fallback)

        unique_cids = sorted(set(sid_to_cid.values()))
        cid_to_smiles = fetch_cid_smiles(unique_cids, batch_size=500)
        for sid, cid in sid_to_cid.items():
            pair = cid_to_smiles.get(cid)
            if pair and pair[0]:
                sid_to_raw_smiles[sid] = pair[0]

    # Compute RDKit representations
    meta, arrays = compute_rdkit_features(sids, sid_to_raw_smiles, n_workers=args.workers)

    # Save sid_smiles.csv
    rows = []
    for sid in sids:
        cid = sid_to_cid.get(sid)
        raw_smi = sid_to_raw_smiles.get(sid, "")
        canon_smi, scaff_smi, _ = meta[sid]
        rows.append(
            {
                "SID": sid,
                "CID": "" if cid is None else cid,
                "SMILES": raw_smi,
                "CanonicalSMILES": canon_smi,
                "ScaffoldSMILES": scaff_smi,
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out} ({len(out_df):,} rows, {csv_out.stat().st_size / 1e6:.2f} MB)", flush=True)

    # Save rdkit_features.npz
    t0 = time.time()
    np.savez_compressed(npz_out, **arrays)
    print(
        f"Saved {npz_out} ({npz_out.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.1f}s",
        flush=True,
    )

    # Verification summary
    valid_sids = {sid for sid, (_, _, ok) in meta.items() if ok}
    no_cid = [sid for sid in sids if sid not in sid_to_cid]
    no_smiles = [sid for sid in sids if sid in sid_to_cid and not sid_to_raw_smiles.get(sid)]
    parse_failed = [sid for sid in sids if sid_to_raw_smiles.get(sid) and sid not in valid_sids]

    unique_canon = {meta[sid][0] for sid in valid_sids if meta[sid][0]}
    unique_scaff = {meta[sid][1] for sid in valid_sids if meta[sid][1]}
    n_nan_desc_cols = int(np.isnan(arrays["rdkit_desc"][arrays["valid_mask"]]).any(axis=0).sum())
    n_all_nan_desc_cols = int(np.isnan(arrays["rdkit_desc"][arrays["valid_mask"]]).all(axis=0).sum())

    print("\n=== VERIFICATION SUMMARY ===", flush=True)
    print(f"Total unique SIDs:           {len(sids):,}")
    print(f"SIDs with CID:               {len(sid_to_cid):,} ({100*len(sid_to_cid)/len(sids):.2f}%)")
    print(f"SIDs with PubChem SMILES:    {len(sid_to_raw_smiles):,} ({100*len(sid_to_raw_smiles)/len(sids):.2f}%)")
    print(f"SIDs with valid RDKit Mol:   {len(valid_sids):,} ({100*len(valid_sids)/len(sids):.2f}%)")
    print(f"  Missing CID:               {len(no_cid)}")
    print(f"  CID found, no SMILES:      {len(no_smiles)}")
    print(f"  SMILES failed RDKit parse: {len(parse_failed)}")
    if parse_failed:
        print(f"  Sample parse failures:     {parse_failed[:10]}")
    print(f"Unique CanonicalSMILES:      {len(unique_canon):,}")
    print(f"Unique Murcko scaffolds:     {len(unique_scaff):,}")
    print(
        f"RDKit 2D descriptors:        {arrays['rdkit_desc'].shape[1]} "
        f"(cols with any NaN among valid mols: {n_nan_desc_cols}, all-NaN: {n_all_nan_desc_cols})"
    )

    print("\nPer-split coverage:")
    for iso in ISOZYMES:
        for prefix in ("training", "test"):
            name = f"{prefix}{iso}"
            ss = split_sids[name]
            n_ok = len(ss & valid_sids)
            print(f"  {name:14s}: {n_ok:5,}/{len(ss):5,} ({100*n_ok/len(ss):6.2f}%)")

    print("\nNPZ array shapes and dtypes:")
    for k, v in arrays.items():
        print(f"  {k:18s}: shape={v.shape}, dtype={v.dtype}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
