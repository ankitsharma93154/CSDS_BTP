# Setup — one-time

## 1. GitHub (code hosting for Kaggle)

Create an **empty** repo on github.com (no README/licence/.gitignore — this repo
already has them). Then, from `C:\Users\ak794\OneDrive\Desktop\BTP`:

```bash
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main
```

If GitHub asks for a password, use a **Personal Access Token** (github.com →
Settings → Developer settings → Personal access tokens → Fine-grained →
repo scope) as the password, or install GitHub CLI (`gh auth login`).

After every change I commit, push with:  `git push`

## 2. Kaggle (free GPU)  —  DONE: account + phone verified

Weekly quota: ~30 GPU-hours, resets Saturday 00:00 UTC. A saved notebook keeps
running up to 12 h even with the tab closed.

## 3. Run Stage 1  (MedMNIST3D — paper Table 4)

1. kaggle.com → **Create → New Notebook**.
2. Right-hand panel → **Session options**:
   - **Accelerator**: GPU T4 x2  (or P100)
   - **Internet**: On
3. In the first cell, paste exactly:

   ```python
   !git clone https://github.com/ankitsharma93154/CSDS_BTP.git repo
   %cd repo
   exec(open("kaggle/stage1_medmnist3d.py").read())
   ```

4. Run it. `RUN_MODE` in the script is `"quick"` — a ~15 min sanity pass over
   2 datasets × 4 models. Check the printed summaries look sane.
5. Edit line 19 of `kaggle/stage1_medmnist3d.py` to `RUN_MODE = "full"` (open it
   from the file browser in the notebook, or just re-`exec` after
   `s = open(...).read().replace('"quick"', '"full"', 1)` ) and **Save Version →
   Save & Run All** to run the full protocol in the background (~2–3 h).
6. When done: **Output** tab → download `stage1_results.zip`. Put it in
   `results/` in the local repo (or attach it here) and I'll build the
   paper-vs-ours comparison tables.

## 4. Push new commits

Whenever I commit changes here:  `git push`   (Kaggle re-clones on next run).
