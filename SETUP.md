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

## 2. Kaggle (free GPU)

1. Sign up at kaggle.com.
2. **Settings → Phone Verification** — required to unlock GPU/internet in notebooks.
3. Create → New Notebook → **Settings**:
   - Accelerator: **GPU T4 x2** (or P100)
   - Internet: **On**
4. Weekly quota: ~30 GPU-hours. It resets Saturday 00:00 UTC. A notebook keeps
   running up to 12 h even if you close the tab ("Save Version → Save & Run All").

## 3. Run Stage 1

In a Kaggle GPU notebook, one cell:

```python
!git clone https://github.com/<USERNAME>/<REPO>.git repo
%cd repo
exec(open("kaggle/stage1_medmnist3d.py").read().replace(
    "<your-username>/<your-repo>", "<USERNAME>/<REPO>"))
```

Set `RUN_MODE = "quick"` in the script first for a ~15 min sanity check, then
`"full"`. Download `stage1_results.zip` from the notebook output when done and
drop it into `results/` locally so I can analyse it.
