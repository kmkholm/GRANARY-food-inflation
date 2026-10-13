"""Figure 1 (v2): five connected stage cards with real-data thumbnails (figure_src/thumbs, made by fig1_thumbs.py),
token strips, equations and a protocol timeline. Writes Figure1.drawio and exports PDF/PNG with draw.io desktop."""
import html, subprocess, os, base64
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); TH = os.path.join(HERE, "thumbs")
cells, nid = [], [2]
FONT = "Helvetica"


def _id():
    i = nid[0]; nid[0] += 1; return i


def box(x, y, w, h, text="", fill="#ffffff", stroke="#333333", font=11, bold=False, fc="#1d2b36", align="center",
        rounded=1, dashed=0, sw=1, extra="", valign="middle", italic=False):
    i = _id()
    fs = (1 if bold else 0) + (2 if italic else 0)
    st = (f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={font};"
          f"fontColor={fc};fontFamily={FONT};dashed={dashed};align={align};verticalAlign={valign};arcSize=6;spacing=3;"
          f"fontStyle={fs};" + extra)
    cells.append(f'<mxCell id="{i}" value="{html.escape(text.replace(chr(10), "<br>"))}" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def text(x, y, w, h, t, font=10, fc="#1d2b36", bold=False, italic=False, align="center"):
    return box(x, y, w, h, t, fill="none", stroke="none", font=font, fc=fc, bold=bold, italic=italic, align=align)


def image(x, y, w, name):
    p = os.path.join(TH, name); im = Image.open(p); hh = w * im.size[1] / im.size[0]
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    i = _id()
    st = f"shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;image=data:image/png,{b64};"
    cells.append(f'<mxCell id="{i}" value="" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{hh:.1f}" as="geometry"/></mxCell>')
    return i, hh


def arrow(x1, y1, x2, y2, color="#1d2b36", w=2.2, dashed=0, label=""):
    i = _id()
    st = (f"endArrow=block;endFill=1;endSize=5;html=1;strokeColor={color};strokeWidth={w};dashed={dashed};fontSize=9;"
          f"fontColor={color};fontFamily={FONT};labelBackgroundColor=#ffffff;")
    cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{st}" edge="1" parent="1">'
                 f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
                 f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')


def badge(x, y, n, color):
    box(x, y, 24, 24, str(n), fill=color, stroke="#ffffff", font=12, bold=True, fc="#ffffff", sw=2, extra="ellipse;", rounded=0)


def card(x, y, w, h, title, color, tint, n):
    box(x, y, w, h, "", fill=tint, stroke=color, sw=1.4, extra="shadow=1;")
    box(x, y, w, 30, title, fill=color, stroke=color, font=12, bold=True, fc="#ffffff", align="left",
        extra="spacingLeft=34;", rounded=1)
    badge(x + 6, y + 3, n, "#1d2b36")


INK = "#1d2b36"
C1, T1 = "#6b5a46", "#faf6f0"      # data
C2, T2 = "#a8681b", "#fdf6ea"      # foundation models
C3, T3 = "#24527a", "#eef4fa"      # adapter
C4, T4 = "#2c7a55", "#edf7f1"      # calibration
C5, T5 = "#6d3a75", "#f7eff8"      # outputs
Y0, H = 20, 470

# ---------- 1 data ----------
x = 20; W = 225
card(x, Y0, W, H, "Public monthly data", C1, T1, 1)
yy = Y0 + 40
for name, lab in (("fao.png", "FAOSTAT food CPI · 198 countries\n12-month inflation, 2000–2026"),
                  ("wfp.png", "WFP retail prices · 863 series\n66 countries, 2008–2026"),
                  ("drivers.png", "12 World Bank global prices\nenergy · fertiliser · grains · oils")):
    _, hh = image(x + 30, yy, W - 60, name)
    text(x + 6, yy + hh - 2, W - 12, 30, lab, font=9.5)
    yy += hh + 34

# ---------- 2 foundation models ----------
x2 = x + W + 30; W2 = 215
card(x2, Y0, W2, H, "Frozen foundation models", C2, T2, 2)
yy = Y0 + 42
for name, lab in (("fan_chronos.png", "Chronos-Bolt (base)"), ("fan_timesfm.png", "TimesFM 2.5 (200 M)")):
    box(x2 + 10, yy, W2 - 20, 150, "", fill="#ffffff", stroke="#e3cfae")
    text(x2 + 10, yy + 4, W2 - 20, 18, lab, font=10.5, bold=True, fc=C2)
    image(x2 + 22, yy + 24, W2 - 44, name)
    text(x2 + 10, yy + 122, W2 - 20, 24, "9 quantiles × H, zero-shot", font=9, italic=True)
    yy += 162
text(x2 + 8, Y0 + H - 64, W2 - 16, 56, "no fine-tuning; context ≤ 512 months;\nexample: Egypt, origin Dec 2021\n(black = observed history);\nall inputs are known at origin o", font=9, fc=C2)

# ---------- 3 adapter ----------
x3 = x2 + W2 + 30; W3 = 345
card(x3, Y0, W3, H, "GRANARY cross-attention adapter", C3, T3, 3)
# forecast tokens strip
text(x3 + 10, Y0 + 38, W3 - 20, 18, "forecast tokens  x₁ … x_H   (one per horizon)", font=9.5, bold=True, fc=C3, align="left")
for k in range(12):
    box(x3 + 14 + k * 26, Y0 + 58, 22, 22, "", fill="#f2c27b" if k % 2 == 0 else "#9cc0e4", stroke="#ffffff", rounded=1)
text(x3 + 10, Y0 + 82, W3 - 20, 30, "xₕ = W_f [qᶜₕ ; qᵀₕ] + eₕ + e_id", font=10.5, italic=True, fc=INK)
# memory strip
text(x3 + 10, Y0 + 112, W3 - 20, 18, "memory tokens  m₀ … m₁₂   (history + 12 drivers)", font=9.5, bold=True, fc=C3, align="left")
box(x3 + 14, Y0 + 132, 22, 22, "", fill="#c9b8a3", stroke="#ffffff")
for k in range(12):
    box(x3 + 40 + k * 24, Y0 + 132, 20, 22, "", fill="#c7a6d0", stroke="#ffffff")
# attention block + heatmap
box(x3 + 10, Y0 + 164, 150, 92, "2 × [ self-attention\nover horizons\n→ cross-attention\nto memory ]\nd = 64, 4 heads", fill="#ffffff", stroke=C3, font=9.5, bold=True, fc=C3)
_, hh = image(x3 + 168, Y0 + 162, 168, "attention.png")
text(x3 + 168, Y0 + 162 + hh - 2, 168, 16, "learned cross-attention (FAOSTAT test)", font=8.5, italic=True)
# head
box(x3 + 10, Y0 + 290, W3 - 20, 104,
    "per-horizon head\nπₕ = weight on Chronos-Bolt · δₕ = shift · ωₕ = log-width\n"
    "q̄ₕ(τ) = πₕ qᶜₕ(τ) + (1 − πₕ) qᵀₕ(τ)\n"
    "q̂ₕ(τ) = q̄ₕ(0.5) + δₕ + [q̄ₕ(τ) − q̄ₕ(0.5)] · exp(ωₕ)",
    fill="#ffffff", stroke=C3, font=9.5, fc=INK)
box(x3 + 10, Y0 + 402, W3 - 20, 56, "pinball loss over τ ∈ {0.1, …, 0.9}\ntrained on origins ≤ 2016 (FAO) / 2017 (WFP)\n0.12–0.13 M parameters · ≈ 30 s per run",
    fill=C3, stroke=C3, font=9, fc="#ffffff")

# ---------- 4 calibration ----------
x4 = x3 + W3 + 30; W4 = 215
card(x4, Y0, W4, H, "Online calibration (ACI)", C4, T4, 4)
text(x4 + 8, Y0 + 40, W4 - 16, 34, "only outcomes observed before o\nare used (origins ≤ o − h − 1)", font=9.5, fc=INK)
# timeline of revealed window
tx = x4 + 14
box(tx, Y0 + 82, 120, 16, "36 revealed origins", fill="#a7d3bb", stroke="#ffffff", font=8, fc=INK)
box(tx + 122, Y0 + 82, 44, 16, "h gap", fill="#e6e6e6", stroke="#ffffff", font=8, fc="#666666")
box(tx + 168, Y0 + 79, 18, 22, "o", fill=C4, stroke="#ffffff", font=10, bold=True, fc="#ffffff")
box(x4 + 10, Y0 + 112, W4 - 20, 92,
    "conformity score\nE = max{q̂(0.1) − Δ, Δ − q̂(0.9)}\nACI update (γ = 0.01, α = 0.2)\nαₒ = αₒ₋₁ + γ (α − missₒ*)",
    fill="#ffffff", stroke=C4, font=9.5, fc=INK)
_, hh = image(x4 + 14, Y0 + 216, W4 - 28, "coverage.png")
text(x4 + 8, Y0 + 216 + hh - 2, W4 - 16, 32, "80 % coverage through time, h = 6\nred: uncalibrated · green: ACI", font=8.5, italic=True)
text(x4 + 8, Y0 + H - 70, W4 - 16, 62, "test: every origin from Jan 2020\n(incl. the 2021–22 surge)\n5 seeds per configuration", font=9, fc=C4)

# ---------- 5 outputs ----------
x5 = x4 + W4 + 30; W5 = 220
card(x5, Y0, W5, H, "Outputs", C5, T5, 5)
text(x5 + 8, Y0 + 40, W5 - 16, 20, "calibrated 80 % interval + median", font=9.5, bold=True, fc=C5)
_, hh = image(x5 + 12, Y0 + 62, W5 - 24, "fan_granary.png")
text(x5 + 8, Y0 + 62 + hh - 2, W5 - 16, 30, "Egypt, origin Dec 2021:\nGRANARY + ACI vs observed", font=8.5, italic=True)
yy = Y0 + 62 + hh + 34
text(x5 + 8, yy, W5 - 16, 20, "driver attribution (occlusion)", font=9.5, bold=True, fc=C5)
_, hh2 = image(x5 + 12, yy + 22, W5 - 24, "attribution.png")
text(x5 + 8, yy + 22 + hh2, W5 - 16, 30, "top contributions to 12-month\nforecasts, Jul 2021 – Dec 2022 (pp)", font=8.5, italic=True)

# ---------- arrows between cards ----------
ym = Y0 + H / 2
for xa, xb in ((x + W, x2), (x2 + W2, x3), (x3 + W3, x4), (x4 + W4, x5)):
    arrow(xa + 2, ym, xb - 2, ym)

# ---------- bottom legend strip ----------
yb = Y0 + H + 14
box(20, yb, x5 + W5 - 20, 34, "", fill="#ffffff", stroke="#d0d0d0", rounded=1)
items = [("#f2c27b", "Chronos-Bolt quantiles"), ("#9cc0e4", "TimesFM quantiles"), ("#c9b8a3", "series history token"),
         ("#c7a6d0", "global-driver tokens"), ("#a7d3bb", "revealed outcomes")]
lx = 40
for col, lab in items:
    box(lx, yb + 10, 14, 14, "", fill=col, stroke="#ffffff")
    text(lx + 18, yb + 7, 170, 20, lab, font=9.5, align="left")
    lx += 205
xml = ('<mxfile host="drawio"><diagram name="Figure 1"><mxGraphModel dx="1400" dy="600" grid="0" gridSize="10" guides="1" '
       f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{x5 + W5 + 20}" pageHeight="{yb + 50}" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
src = os.path.join(HERE, "Figure1.drawio"); open(src, "w", encoding="utf-8").write(xml)
fig = os.path.join(HERE, "..", "figures"); os.makedirs(fig, exist_ok=True)
exe = r"C:\Program Files\draw.io\draw.io.exe"
for fmt, extra in (("pdf", ["--crop"]), ("png", ["--scale", "3"])):
    subprocess.run([exe, "-x", "-f", fmt, *extra, "-o", os.path.join(fig, f"Figure1.{fmt}"), src], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("written", src)
