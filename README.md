# Cytochrome P450 Inhibitor Classification with Statistical Learning

M.S. thesis (North Carolina Central University, 2015). LaTeX source is in `thesis/`, analysis notebooks in `notebooks/`.

## Re-running the Python models

1. Download the balanced training and test sets from Figshare into `notebooks/data/`:
   - <http://dx.doi.org/10.6084/m9.figshare.1181846>
   - <http://dx.doi.org/10.6084/m9.figshare.1066108>

   The notebooks expect `training{1a2,2c9,2c19,2d6,3a4}.csv` and `test{...}.csv`.
2. Execute the notebooks:

   ```sh
   uv sync
   cd notebooks
   uv run jupyter nbconvert --to notebook --execute --inplace Model*.ipynb
   ```

Results are written to `notebooks/results/<isozyme>.csv`. Each model is cross-validated on the training set and then fit on the full training set to predict the held-out test set (`notebooks/cyp_models.py`).
