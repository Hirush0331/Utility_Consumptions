# Utility Prediction Model — Fix Summary (2026-07-08)

## Files in this package
- `electricity_pkl.sav`, `steam_pkl.sav`, `water_pkl.sav` — retrained models (bug fixed)
- `app.py` — Streamlit app (pickle load paths fixed: no more `Consumptions/` prefix)
- `requirements.txt` — pinned to the exact versions used to train these models
- `train_models.py` — new single-script trainer to replace the 3 manual notebooks
- `Consumptions.xlsx` — the dataset used for this training run (834 rows, 2024-01-01 to 2026-06-09)

## The bug that was found
In `Electricity.ipynb`, `Steam.ipynb`, and `Water.ipynb`, the outlier-handling step
replaced "outlier" machine-count **inputs** with the column median (IQR rule), but
did **not** apply the same treatment to the **target** (Electricity/Steam/Water
values). Up to 21% of rows per column were affected.

Result: on real high-activity days, the model was trained on
`typical machine counts -> unusually high consumption`, teaching it a false
relationship. This is what caused the systematic over-prediction you measured
(~+10-13% on Electricity and Steam).

## What changed in `train_models.py`
- Removed the broken outlier-replacement step entirely — real production
  variation is now kept as-is, matched correctly to its real output value.
- Kept everything else identical to your original approach: same 18 features,
  same 80/20 split (`random_state=42`), same 3-model stack
  (ExtraTrees + GradientBoosting + DecisionTree → Ridge meta-learner).
- Added MAPE reporting alongside R²/MAE/RMSE so future accuracy checks are
  easier to interpret.

## Retrained model performance (on this dataset)
| Target | R² | MAE | MAPE |
|---|---|---|---|
| Electricity (kWh) | 0.940 | 1,126 | 10.2% |
| Steam (kg) | 0.932 | 3,873 | 22.8% |
| Water (Cu.m.) | 0.923 | 66.5 | 9.3% |

Steam's MAPE looks high mainly because MAPE inflates when actual values are
small (a few low-consumption days dominate the % error even though the
absolute error is reasonable). R² for Steam (0.932) is actually strong — worth
keeping an eye on but not alarming.

## Going forward: how to retrain
1. Update `Consumptions.xlsx` with new rows.
2. Run:
   ```bash
   python train_models.py
   ```
   This reads the Excel file, retrains all 3 models, prints metrics, and
   overwrites the 3 `.sav` files — one command instead of running 3 notebooks.
3. Commit and push the updated `.sav` files (and `Consumptions.xlsx` if you
   want it versioned) to GitHub.
4. Reboot the app on Streamlit Cloud.

## Deploy steps for this specific update
```bash
git add electricity_pkl.sav steam_pkl.sav water_pkl.sav app.py requirements.txt train_models.py
git commit -m "Fix outlier-handling bug causing prediction bias; retrain models"
git push origin main
```
Then on share.streamlit.io: **⋮ menu → Reboot app**.
