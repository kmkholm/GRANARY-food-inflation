"""Check float placement: for every figure/table, the page of its caption vs the page of its first mention in the text.
python check_floats.py <pdf>"""
import re, subprocess, sys

pdf = sys.argv[1]
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
pages = [subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), pdf, "-"], capture_output=True, text=True,
                        encoding="utf-8", errors="ignore").stdout for p in range(1, n + 1)]
rows = []
for kind in ("Figure", "Table"):
    for k in range(1, 13):
        cap = [i + 1 for i, t in enumerate(pages) if re.search(rf"{kind} {k}:", t)]
        if not cap:
            continue
        # first mention that is not the caption itself
        men = [i + 1 for i, t in enumerate(pages) if re.search(rf"{kind}s? {k}(?![:\d])", t.replace(f"{kind} {k}:", ""))]
        first = min(men) if men else None
        rows.append((kind, k, cap[0], first, (cap[0] - first) if first else None))
for r in rows:
    flag = "" if r[4] is not None and 0 <= r[4] <= 1 else "  <-- check"
    print(f"{r[0]} {r[1]:2d}: caption p{r[2]:3d}  first mention p{r[3]}  gap {r[4]}{flag}")
