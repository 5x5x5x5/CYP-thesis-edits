"""Compute deterministic RDKit ETKDGv3 3D conformer descriptors (Shape + PBF + USRCAT + AUTOCORR3D)
for all 16,725 unique PubChem SIDs in `notebooks/data/sid_smiles.csv`.

Saves `notebooks/data/rdkit_3d_features.npz` with keys:
  - sids: (16725,) int64
  - valid_3d: (16725,) bool
  - desc_3d: (16725, 151) float32 (NaN where 3D embedding failed or SMILES missing)
  - desc_3d_names: (151,) str
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).resolve().parent / "data"
N_WORKERS = 4
SHAPE_FUNCS = [
    ("Asphericity", rdMolDescriptors.CalcAsphericity),
    ("Eccentricity", rdMolDescriptors.CalcEccentricity),
    ("InertialShapeFactor", rdMolDescriptors.CalcInertialShapeFactor),
    ("NPR1", rdMolDescriptors.CalcNPR1),
    ("NPR2", rdMolDescriptors.CalcNPR2),
    ("PMI1", rdMolDescriptors.CalcPMI1),
    ("PMI2", rdMolDescriptors.CalcPMI2),
    ("PMI3", rdMolDescriptors.CalcPMI3),
    ("RadiusOfGyration", rdMolDescriptors.CalcRadiusOfGyration),
    ("SpherocityIndex", rdMolDescriptors.CalcSpherocityIndex),
    ("PBF", rdMolDescriptors.CalcPBF),
]
DIM_3D = len(SHAPE_FUNCS) + 60 + 80  # 11 shape + 60 USRCAT + 80 AUTOCORR3D = 151


def featurize_one_3d(smiles: str) -> tuple[bool, np.ndarray]:
    out = np.full(DIM_3D, np.nan, dtype=np.float32)
    if not isinstance(smiles, str) or not smiles.strip():
        return False, out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return False, out
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        params.useRandomCoords = False
        params.maxIterations = 200
        res = AllChem.EmbedMolecule(mol_h, params)
        if res != 0:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol_h, params)
        if res != 0:
            return False, out

        vals: list[float] = []
        for _, fn in SHAPE_FUNCS:
            vals.append(float(fn(mol_h)))
        vals.extend(float(v) for v in rdMolDescriptors.GetUSRCAT(mol_h))
        vals.extend(float(v) for v in rdMolDescriptors.CalcAUTOCORR3D(mol_h))
        arr = np.array(vals, dtype=np.float32)
        arr = np.where(np.isfinite(arr), np.clip(arr, -1e6, 1e6), np.nan)
        return True, arr
    except Exception:
        return False, out


def main() -> None:
    t0 = time.time()
    sm = pd.read_csv(DATA_DIR / "sid_smiles.csv", usecols=["SID", "CanonicalSMILES", "SMILES"])
    sids = sm["SID"].to_numpy(dtype=np.int64)
    smiles_list = [
        cs.strip() if isinstance(cs, str) and cs.strip() else (s.strip() if isinstance(s, str) else "")
        for cs, s in zip(sm["CanonicalSMILES"], sm["SMILES"])
    ]
    print(f"Generating ETKDGv3 3D conformers + 151 3D descriptors for {len(sids)} molecules ({N_WORKERS} workers)...", flush=True)

    valid = np.zeros(len(sids), dtype=bool)
    desc = np.full((len(sids), DIM_3D), np.nan, dtype=np.float32)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (ok, vec) in enumerate(ex.map(featurize_one_3d, smiles_list, chunksize=64)):
            valid[i] = ok
            desc[i] = vec
            if (i + 1) % 3000 == 0:
                print(f"  processed {i + 1}/{len(sids)} ({valid[:i+1].sum()} valid 3D) in {time.time() - t0:.1f}s", flush=True)

    names = (
        [n for n, _ in SHAPE_FUNCS]
        + [f"USRCAT_{k}" for k in range(60)]
        + [f"AUTOCORR3D_{k}" for k in range(80)]
    )
    out_path = DATA_DIR / "rdkit_3d_features.npz"
    np.savez_compressed(
        out_path,
        sids=sids,
        valid_3d=valid,
        desc_3d=desc,
        desc_3d_names=np.array(names),
    )
    print(
        f"Done in {time.time() - t0:.1f}s: {valid.sum()}/{len(sids)} ({100*valid.mean():.2f}%) valid 3D conformers -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
