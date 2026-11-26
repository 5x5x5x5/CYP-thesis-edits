"""Stage #3 (Tier 5C): Targeted Boltz-2 All-Atom Co-Folding [CYP + Heme (HEM) + Ligand]
on the Top 300 Novel-Scaffold / High-Uncertainty Held-Out Test Compounds (1A2, 2D6, 3A4).

Strategy for maximum speed and zero wasted compute:
  1. Uses exact canonical human UniProt sequences (P05177=1A2, P10635=2D6, P08684=3A4),
     explicit Heme-b cofactor (`ccd: HEM`), and each test compound's CanonicalSMILES,
     with Boltz-2's native `properties: [affinity: {binder: L}]` prediction head enabled.
  2. Runs one initial complex per isozyme with `--use_msa_server` to cache the `.a3m`
     MSA for CYP 1A2, 2D6, and 3A4 ONCE, then reuses that exact `.a3m` path across all
     remaining 99 ligands of that isozyme (5x faster, zero redundant MSA server calls!).
  3. Extracts from every predicted complex:
     - `affinity_probability_binary` (Boltz-2 binding probability)
     - `affinity_pred_value` (Boltz-2 log10 IC50 / potency proxy)
     - `confidence_score`, `iptm`, `ligand_iptm`, `complex_plddt`
     - Exact 3D minimum distance (`d_fe_min`, Å) from any ligand heteroatom (N, O, S)
       to the Heme iron (`FE`) atom in the predicted `.cif` structure (Type II ligation!).
  4. Saves incremental CSV after every batch to `boltz2_top300_results.csv`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CYP_SEQS = {
    "1a2": (
        "MALSQSVPFSATELLLASAIFCLVFWVLKGLRPRVPKGLKSPPEPWGWPLLGHVLTLGKN"
        "PHLALSRMSQRYGDVLQIRIGSTPVLVLSRLDTIRQALVRQGDDFKGRPDLYTSTLITDG"
        "QSLTFSTDSGPVWAARRRLAQNALNTFSIASDPASSSSCYLEEHVSKEAKALISRLQELM"
        "AGPGHFDPYNQVVVSVANVIGAMCFGQHFPESSDEMLSLVKNTHEFVETASSGNPLDFFP"
        "ILRYLPNPALQRFKAFNQRFLWFLQKTVQEHYQDFDKNSVRDITGALFKHSKKGPRASGN"
        "LIPQEKIVNLVNDIFGAGFDTVTTAISWSLMYLVTKPEIQRKIQKELDTVIGRERRPRLS"
        "DRPQLPYLEAFILETFRHSSFLPFTIPHSTTRDTTLNGFYIPKKCCVFVNQWQVNHDPEL"
        "WEDPSEFRPERFLTADGTAINKPLSEKMMLFGMGKRRCIGEVLAKWEIFLFLAILLQQLE"
        "FSVPPGVKVDLTPIYGLTMKHARCEHVQARLRFSIN"
    ),
    "2d6": (
        "MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQ"
        "LRRRFGDVFSLQLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVF"
        "LARYGPAWREQRRFSVSTLRNLGLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDK"
        "AVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGLKEESGFLREVLNAVPVLLHIPALAGKV"
        "LRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAEMEKAKGNPESSFNDENLRIVVA"
        "DLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPYTTAVI"
        "HEVQRFGDIVPLGVTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWEKPFRFHPEHF"
        "LDAQGHFVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGV"
        "FAFLVSPSPYELCAVPR"
    ),
    "3a4": (
        "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMF"
        "DMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISI"
        "AEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYS"
        "MDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICV"
        "FPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSI"
        "IFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVV"
        "NETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFS"
        "KKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLG"
        "GLLQPEKPVVLKVESRDGTVSGA"
    ),
}


def write_boltz_yaml(
    out_yaml: Path, iso: str, smiles: str, msa_path: Path | None = None
) -> None:
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {CYP_SEQS[iso]}",
    ]
    if msa_path is not None and msa_path.exists():
        lines.append(f"      msa: {msa_path}")
    lines.extend(
        [
            "  - ligand:",
            "      id: H",
            "      ccd: HEM",
            "  - ligand:",
            "      id: L",
            f"      smiles: '{smiles}'",
            "properties:",
            "  - affinity:",
            "      binder: L",
        ]
    )
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_fe_distances(cif_path: Path) -> tuple[float, float]:
    """Return (min_het_to_Fe_A, min_any_ligand_to_Fe_A) from a Boltz-2 mmCIF file."""
    if not cif_path.exists():
        return float("nan"), float("nan")
    fe_coords = []
    lig_het = []
    lig_all = []
    cols: list[str] = []
    in_atom_loop = False
    for line in cif_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            if not in_atom_loop:
                cols = []
                in_atom_loop = True
            cols.append(s.split(".")[1])
            continue
        if in_atom_loop and (s.startswith("ATOM ") or s.startswith("HETATM ")):
            parts = s.split()
            if len(parts) < len(cols):
                continue
            rec = dict(zip(cols, parts))
            chain = rec.get("auth_asym_id") or rec.get("label_asym_id", "")
            elem = (rec.get("type_symbol") or "").upper()
            try:
                xyz = np.array(
                    [float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"])],
                    dtype=np.float64,
                )
            except (KeyError, ValueError):
                continue
            if chain == "H" and elem == "FE":
                fe_coords.append(xyz)
            elif chain == "L" and elem != "H":
                lig_all.append(xyz)
                if elem in ("N", "O", "S"):
                    lig_het.append(xyz)
        elif in_atom_loop and s and not s.startswith("#"):
            in_atom_loop = False

    if not fe_coords or not lig_all:
        return float("nan"), float("nan")
    fe = fe_coords[0]
    d_all = float(min(np.linalg.norm(p - fe) for p in lig_all))
    d_het = float(min(np.linalg.norm(p - fe) for p in lig_het)) if lig_het else float("nan")
    return d_het, d_all


def extract_record(pred_dir: Path, name: str, row: pd.Series) -> dict:
    rec = {
        "isozyme": row["isozyme"],
        "SID": int(row["SID"]),
        "y_test": int(row["y_test"]),
        "ActivityScore": float(row["ActivityScore"]),
        "is_novel_scaffold": bool(row["is_novel_scaffold"]),
        "grand_prob": float(row["grand_prob"]),
        "affinity_prob": float("nan"),
        "affinity_val": float("nan"),
        "confidence_score": float("nan"),
        "iptm": float("nan"),
        "ligand_iptm": float("nan"),
        "d_fe_het": float("nan"),
        "d_fe_all": float("nan"),
    }
    aff_file = pred_dir / f"affinity_{name}.json"
    if aff_file.exists():
        aj = json.loads(aff_file.read_text())
        rec["affinity_prob"] = float(aj.get("affinity_probability_binary", float("nan")))
        rec["affinity_val"] = float(aj.get("affinity_pred_value", float("nan")))

    conf_file = pred_dir / f"confidence_{name}_model_0.json"
    if conf_file.exists():
        cj = json.loads(conf_file.read_text())
        rec["confidence_score"] = float(cj.get("confidence_score", float("nan")))
        rec["iptm"] = float(cj.get("iptm", float("nan")))
        rec["ligand_iptm"] = float(cj.get("ligand_iptm", float("nan")))

    cif_file = pred_dir / f"{name}_model_0.cif"
    d_het, d_all = parse_fe_distances(cif_file)
    rec["d_fe_het"] = d_het
    rec["d_fe_all"] = d_all
    return rec


def run_boltz_batch(
    yaml_dir: Path, out_dir: Path, use_msa_server: bool, sampling_steps: int = 50
) -> None:
    cmd = [
        "boltz",
        "predict",
        str(yaml_dir),
        "--out_dir",
        str(out_dir),
        "--diffusion_samples",
        "1",
        "--recycling_steps",
        "3",
        "--sampling_steps",
        str(sampling_steps),
        "--sampling_steps_affinity",
        "50",
        "--diffusion_samples_affinity",
        "1",
        "--override",
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    print(f"Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"WARNING: boltz predict exited with code {res.returncode}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/boltz_work"))
    parser.add_argument("--out-csv", type=Path, default=Path("/content/results/boltz2_top300_results.csv"))
    parser.add_argument("--isozymes", nargs="+", default=["1a2", "2d6", "3a4"])
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_recs: list[dict] = []
    if args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        all_recs = prev.to_dict(orient="records")
        done_keys = {(r["isozyme"], int(r["SID"])) for r in all_recs}
    else:
        done_keys = set()

    t0 = time.time()
    for iso in args.isozymes:
        sub = df[df["isozyme"] == iso].reset_index(drop=True)
        print(f"\n{'='*72}\nBoltz-2 + Heme: CYP {iso.upper()} ({len(sub)} compounds)\n{'='*72}", flush=True)

        # Step 1: Run first compound with --use_msa_server to generate & cache the isozyme MSA once
        msa_cache_dir = args.work_dir / f"msa_seed_{iso}"
        msa_out_dir = args.work_dir / f"out_seed_{iso}"
        msa_cache_dir.mkdir(parents=True, exist_ok=True)
        first = sub.iloc[0]
        first_name = f"{iso}_{int(first['SID'])}"
        write_boltz_yaml(msa_cache_dir / f"{first_name}.yaml", iso, first["CanonicalSMILES"], msa_path=None)

        msa_files = list(msa_out_dir.rglob("*.a3m")) if msa_out_dir.exists() else []
        if not msa_files:
            run_boltz_batch(msa_cache_dir, msa_out_dir, use_msa_server=True)
            msa_files = list(msa_out_dir.rglob("*.a3m"))

        msa_path = msa_files[0] if msa_files else None
        print(f"  [{iso}] Cached MSA: {msa_path}", flush=True)

        if (iso, int(first["SID"])) not in done_keys:
            pdir = msa_out_dir / f"boltz_results_{msa_cache_dir.name}" / "predictions" / first_name
            rec = extract_record(pdir, first_name, first)
            all_recs.append(rec)
            done_keys.add((iso, int(first["SID"])))
            pd.DataFrame(all_recs).to_csv(args.out_csv, index=False)

        # Step 2: Run remaining compounds in batches reusing cached MSA
        rem = [sub.iloc[i] for i in range(1, len(sub)) if (iso, int(sub.iloc[i]["SID"])) not in done_keys]
        for b_start in range(0, len(rem), args.batch_size):
            batch = rem[b_start : b_start + args.batch_size]
            b_idx = b_start // args.batch_size
            b_yaml = args.work_dir / f"batch_{iso}_{b_idx}"
            b_out = args.work_dir / f"out_{iso}_{b_idx}"
            b_yaml.mkdir(parents=True, exist_ok=True)
            for r in batch:
                nm = f"{iso}_{int(r['SID'])}"
                write_boltz_yaml(b_yaml / f"{nm}.yaml", iso, r["CanonicalSMILES"], msa_path=msa_path)

            t_b = time.time()
            run_boltz_batch(b_yaml, b_out, use_msa_server=(msa_path is None))
            for r in batch:
                nm = f"{iso}_{int(r['SID'])}"
                pdir = b_out / f"boltz_results_{b_yaml.name}" / "predictions" / nm
                rec = extract_record(pdir, nm, r)
                all_recs.append(rec)
                done_keys.add((iso, int(r["SID"])))
            pd.DataFrame(all_recs).to_csv(args.out_csv, index=False)
            n_ok = sum(np.isfinite([r["affinity_prob"] for r in all_recs]))
            print(
                f"  [{iso}] Batch {b_idx} ({len(batch)} ligs) done in {time.time()-t_b:.1f}s | "
                f"Total saved={len(all_recs)} (valid aff={n_ok}) | Elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    print(f"\nBoltz-2 Top-300 complete in {(time.time()-t0)/60:.2f} min -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
