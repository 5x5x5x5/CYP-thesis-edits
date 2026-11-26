"""Tier 5 Grand Meta-Stack: Incorporates Stage #1/#2 Frontier Models
(`MT_Joint3D_OpenSource` & `MT_Joint3D_AllFeatures`: 2D FPs + 151 3D ETKDGv3 Conformer
Descriptors + 768-d 1.1B-Molecule MoLFormer-XL Embeddings + Joint Binary & Continuous
ActivityScore Multi-Task Training) into the nested OOF meta-stacker and runs exact
DeLong & McNemar significance tests against Tier 4 (0.8792) and 2015 baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tier4_grand_ensemble import (
    ALL_MODAL_MEMBERS,
    ISOZYMES,
    N_SPLITS,
    OPEN_SOURCE_MEMBERS,
    RESULTS_DIR,
    SEED,
    compute_metrics,
    delong_roc_test,
    load_isozyme_predictions,
    mcnemar_test,
    run_nested_simplex_blend,
    run_nested_stack,
)

T5_PREDS_DIR = RESULTS_DIR / "tier5_grand_preds"
T5_OPEN_MEMBERS = OPEN_SOURCE_MEMBERS + ["MT_Joint3D_OpenSource"]
T5_ALL_MEMBERS = ALL_MODAL_MEMBERS + ["MT_Joint3D_OpenSource", "MT_Joint3D_AllFeatures"]


def main() -> None:
    T5_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stat_rows: list[dict] = []

    for iso in ISOZYMES:
        y_tr, y_te, sid_tr, sid_te, oof_d, test_d = load_isozyme_predictions(iso)
        # Also load Tier 4 grand preds and Tier 5 joint preds
        for sub in ["grand_preds", "tier5_joint_preds"]:
            p = RESULTS_DIR / sub / f"{iso}.npz"
            if p.exists():
                z = np.load(p)
                assert np.array_equal(sid_tr, z["sid_train"])
                assert np.array_equal(sid_te, z["sid_test"])
                for k in z.files:
                    if k.startswith("oof_"):
                        oof_d[k[4:]] = z[k].astype(np.float64)
                    elif k.startswith("test_"):
                        test_d[k[5:]] = z[k].astype(np.float64)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(skf.split(np.zeros(len(y_tr)), y_tr))

        os_oof, os_te, os_p = run_nested_stack(T5_OPEN_MEMBERS, oof_d, test_d, y_tr, folds)
        all_oof, all_te, all_p = run_nested_stack(T5_ALL_MEMBERS, oof_d, test_d, y_tr, folds)
        bl_oof, bl_te, bl_p = run_nested_simplex_blend(T5_ALL_MEMBERS, oof_d, test_d, y_tr, folds)

        oof_d["Tier5_OpenSource_Stack"] = os_oof
        test_d["Tier5_OpenSource_Stack"] = os_te
        oof_d["Tier5_AllModal_Stack"] = all_oof
        test_d["Tier5_AllModal_Stack"] = all_te
        oof_d["Tier5_AllModal_Blend"] = bl_oof
        test_d["Tier5_AllModal_Blend"] = bl_te

        for fset, mname, oof, tp, params in [
            ("OpenSource_Tier5", "Tier5_OpenSource_Stack", os_oof, os_te, os_p),
            ("AllModal_Tier5", "Tier5_AllModal_Stack", all_oof, all_te, all_p),
            ("AllModal_Tier5", "Tier5_AllModal_Blend", bl_oof, bl_te, bl_p),
        ]:
            m = compute_metrics(y_tr, oof, y_te, tp)
            rows.append(
                {"isozyme": iso, "feature_set": fset, "model": mname, **m, "best_params": json.dumps(params, sort_keys=True)}
            )

        for ma, mb, label in [
            ("Grand_AllModal_Stack", "Tier5_AllModal_Stack", "Tier5_vs_Tier4_AllModal"),
            ("Grand_OpenSource_Stack", "Tier5_OpenSource_Stack", "Tier5_vs_Tier4_OpenSource"),
            ("Tier5_OpenSource_Stack", "Tier5_AllModal_Stack", "Tier5_AllModal_vs_Tier5_OpenSource"),
            ("SVM_baseline", "Tier5_AllModal_Stack", "Tier5_AllModal_vs_OrigSVM"),
        ]:
            if ma in test_d and mb in test_d:
                dl = delong_roc_test(y_te, test_d[ma], test_d[mb])
                mc = mcnemar_test(y_te, test_d[ma], test_d[mb])
                stat_rows.append({"isozyme": iso, "comparison": label, "model_a": ma, "model_b": mb, **dl, **mc})

        np.savez_compressed(
            T5_PREDS_DIR / f"{iso}.npz",
            y_train=y_tr,
            y_test=y_te,
            sid_train=sid_tr,
            sid_test=sid_te,
            oof_Tier5_OpenSource_Stack=os_oof,
            test_Tier5_OpenSource_Stack=os_te,
            oof_Tier5_AllModal_Stack=all_oof,
            test_Tier5_AllModal_Stack=all_te,
            oof_Tier5_AllModal_Blend=bl_oof,
            test_Tier5_AllModal_Blend=bl_te,
        )

    df_r = pd.DataFrame(rows)
    df_s = pd.DataFrame(stat_rows)
    df_r.to_csv(RESULTS_DIR / "tier5_grand_summary.csv", index=False)
    df_s.to_csv(RESULTS_DIR / "tier5_statistical_tests.csv", index=False)

    for col in ["test_auc", "test_acc", "test_brier", "test_ece"]:
        p = df_r.pivot(index="model", columns="isozyme", values=col)[ISOZYMES]
        p["mean"] = p.mean(axis=1)
        print(f"\n-- {col} --")
        print(p.round(4).to_string())

    print("\n=== TIER 5 DELONG TESTS ===")
    for comp, grp in df_s.groupby("comparison", sort=False):
        print(f"\n[{comp}]")
        print(grp[["isozyme", "auc_a", "auc_b", "delta_auc", "delong_p", "delta_acc", "mcnemar_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
