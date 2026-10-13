"""Fixed grid (2026-10-13 14:20): 2 datasets x 5 seeds x 6 variants = 60 runs, ~40 min. Skips finished runs."""
import os, subprocess, sys, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__))
V = {"full": [], "nodrv": ["--no_drivers"], "nohist": ["--no_history"], "noid": ["--no_identity"],
     "onlychronos": ["--single", "chronos"], "onlytimesfm": ["--single", "timesfm"]}
suffix = {"full": "", "nodrv": "_nodrv", "nohist": "_nohist", "noid": "_noid", "onlychronos": "_onlychronos", "onlytimesfm": "_onlytimesfm"}
log = open(os.path.join(HERE, "..", "runs", "grid_progress.txt"), "a")
for ds, seed, v in itertools.product(["fao", "wfp"], range(5), V):
    out = os.path.join(HERE, "..", "runs", "main", f"{ds}_s{seed}{suffix[v]}", "result.json")
    if os.path.exists(out):
        continue
    t = time.time()
    rc = subprocess.call([sys.executable, os.path.join(HERE, "granary.py"), "--ds", ds, "--seed", str(seed), "--epochs", "25", "--tag", "main"] + V[v],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
    log.write(f"{time.strftime('%H:%M:%S')} rc={rc} {ds} s{seed} {v} {time.time()-t:.0f}s\n"); log.flush()
log.write("GRID DONE\n")
