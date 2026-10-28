"""Generate the SENSE-GPT architecture figure: four inline-SVG panels in an HTML page + standalone SVGs for the paper."""
from pathlib import Path
import html

OUT_HTML = Path("/tmp/claude-1014/-fast-lev-TRUST-MRI/c112abab-9060-45be-af69-4cdb48cf240e/scratchpad/architecture.html")
OUT_DIR = Path("/fast/lev/TRUST-MRI/paper_figs")

LIGHT = dict(bg="#fbfaf7", ink="#1f2328", muted="#6b7075", train="#dbe7f3", train_s="#3b6ea5", frozen="#e9e8e4", frozen_s="#8a8f94",
             new="#fbe9c6", new_s="#b7791f", accent="#3b6ea5")
FONT = "'IBM Plex Sans', system-ui, sans-serif"


class Canvas:
    def __init__(self, W, H):
        self.W, self.H, self.items = W, H, []

    def _c(self, key, standalone):
        return LIGHT[key] if standalone else f"var(--{key})"

    def box(self, x, y, w, h, lines, kind="plain", fs=12, dashed=False):
        self.items.append(("box", x, y, w, h, lines, kind, fs, dashed))

    def arrow(self, x1, y1, x2, y2, label=None, dashed=False, lx=None, ly=None, fs=11, anchor="middle", color="ink"):
        self.items.append(("arrow", x1, y1, x2, y2, label, dashed, lx, ly, fs, anchor, color))

    def line(self, pts, label=None, dashed=False, lx=None, ly=None, fs=11, arrow=True, color="ink"):
        self.items.append(("line", pts, label, dashed, lx, ly, fs, arrow, color))

    def text(self, x, y, s, fs=12, anchor="middle", bold=False, italic=False, color="ink"):
        self.items.append(("text", x, y, s, fs, anchor, bold, italic, color))

    def plus(self, x, y, r=10):
        self.items.append(("plus", x, y, r))

    def region(self, x, y, w, h, title=None):
        self.items.append(("region", x, y, w, h, title))

    def _label(self, lx, ly, label, fs, anchor, c):
        tw = 0.56 * fs * len(label) + 6
        x0 = {"middle": lx - tw / 2, "start": lx - 3, "end": lx - tw + 3}[anchor]
        return (f'<rect x="{x0:.1f}" y="{ly - fs + 1:.1f}" width="{tw:.1f}" height="{fs + 4}" fill="{c("bg")}"/>'
                f'<text x="{lx}" y="{ly}" font-size="{fs}" text-anchor="{anchor}" fill="{c("ink")}">{html.escape(label)}</text>')

    def render(self, standalone=False, label=""):
        c = lambda k: self._c(k, standalone)
        o = []
        if standalone:
            o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" width="{self.W}" height="{self.H}" '
                     f'font-family="{FONT}" style="background:{LIGHT["bg"]};color:{LIGHT["ink"]}">')
        else:
            o.append(f'<svg viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{html.escape(label)}" font-family="{FONT}">')
        o.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 f'<path d="M0,0 L10,5 L0,10 z" fill="{c("ink")}"/></marker>'
                 f'<marker id="ahm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                 f'<path d="M0,0 L10,5 L0,10 z" fill="{c("muted")}"/></marker></defs>')
        fills = {"plain": (c("bg"), c("ink")), "train": (c("train"), c("train_s")), "frozen": (c("frozen"), c("frozen_s")),
                 "new": (c("new"), c("new_s")), "infer": (c("bg"), c("ink"))}
        for it in self.items:
            k = it[0]
            if k == "region":
                _, x, y, w, h, title = it
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{c("muted")}" stroke-width="0.8" stroke-dasharray="3 3"/>')
                if title:
                    o.append(f'<text x="{x + w - 8}" y="{y + 15}" font-size="11.5" font-weight="600" text-anchor="end" fill="{c("muted")}" letter-spacing="0.4">{html.escape(title)}</text>')
            elif k == "box":
                _, x, y, w, h, lines, kind, fs, dashed = it
                fill, stroke = fills[kind]
                dash = ' stroke-dasharray="5 3"' if (dashed or kind == "infer") else ""
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.1"{dash}/>')
                n = len(lines); lh = fs + 3
                if 0.62 * fs * len(lines[0]) > w - 8:
                    print("WARN narrow box:", lines[0], "needs ~", int(0.62 * fs * len(lines[0]) + 8), "has", w)
                y0 = y + h / 2 - (n - 1) * lh / 2 + fs * 0.35
                for i, ln in enumerate(lines):
                    bold = i == 0 and n > 1
                    bw = ' font-weight="600"' if bold else ''
                    o.append(f'<text x="{x + w / 2}" y="{y0 + i * lh:.1f}" font-size="{fs if bold else fs - 1}" text-anchor="middle" '
                             f'fill="{c("ink")}"{bw}>{html.escape(ln)}</text>')
            elif k == "arrow":
                _, x1, y1, x2, y2, label, dashed, lx, ly, fs, anchor, color = it
                dash = ' stroke-dasharray="5 3"' if dashed else ""
                mk = "ahm" if color == "muted" else "ah"
                o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c(color)}" stroke-width="1.1" marker-end="url(#{mk})"{dash}/>')
                if label:
                    lx = (x1 + x2) / 2 if lx is None else lx; ly = (y1 + y2) / 2 - 6 if ly is None else ly
                    o.append(self._label(lx, ly, label, fs, anchor, c))
            elif k == "line":
                _, pts, label, dashed, lx, ly, fs, arrow, color = it
                dash = ' stroke-dasharray="5 3"' if dashed else ""
                mk = f' marker-end="url(#{"ahm" if color == "muted" else "ah"})"' if arrow else ""
                o.append(f'<polyline points="{" ".join(f"{p[0]},{p[1]}" for p in pts)}" fill="none" stroke="{c(color)}" stroke-width="1.1"{mk}{dash}/>')
                if label:
                    o.append(self._label(lx, ly, label, fs, "middle", c))
            elif k == "text":
                _, x, y, s, fs, anchor, bold, italic, color = it
                bw = ' font-weight="600"' if bold else ''
                iw = ' font-style="italic"' if italic else ''
                o.append(f'<text x="{x}" y="{y}" font-size="{fs}" text-anchor="{anchor}" fill="{c(color)}"{bw}{iw}>{html.escape(s)}</text>')
            elif k == "plus":
                _, x, y, r = it
                o.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c("bg")}" stroke="{c("ink")}" stroke-width="1.1"/>'
                         f'<text x="{x}" y="{y + 4.5}" font-size="14" text-anchor="middle" fill="{c("ink")}">+</text>')
        o.append("</svg>")
        return "\n".join(o)


# ---------------------------------------------------------------- Panel A: pipeline
def panel_a():
    cv = Canvas(1320, 360)
    y, h = 110, 58
    xs = [20, 200, 380, 560, 740, 920, 1100]; w = 160
    cv.box(xs[0], y, w, h, ["undersampled k-space", "y, C coils, mask M", "R = 8, 10 ACS lines"])
    cv.box(xs[1], y, w, h, ["sensitivity net", "U-Net on the ACS lines", "→ maps S"], "train")
    cv.box(xs[2], y, w, h, ["SENSE combine", "x_zf = Σ S*·F⁻¹y / Σ|S|²", "256 × 256, complex"])
    cv.box(xs[3], y, w, h, ["MedITok encoder", "frozen, Panel B", "Re and Im separately"], "frozen")
    cv.box(xs[4], y, w, h, ["GPT-L body + heads", "Panels C, D", "426 M, trainable"], "train")
    cv.box(xs[5], y, w, h, ["MedITok decoder", "frozen, Panel B"], "frozen")
    cv.box(xs[6], y, 185, h, ["expand x̂_c = S_c · x̂", "per-coil DC on the", "measured lines"], "infer")
    ym = y + h / 2
    cv.arrow(xs[0] + w, ym, xs[1], ym, "y")
    cv.arrow(xs[1] + w, ym, xs[2], ym, "S, y")
    cv.arrow(xs[2] + w, ym, xs[3], ym, "x_zf, ↑2 → 512²", ly=ym - 9)
    cv.arrow(xs[3] + w, ym, xs[4], ym, "z_re, z_im", ly=ym - 9)
    cv.text(xs[4] - 15, ym + 22, "32×32×64 each", fs=10.5)
    cv.arrow(xs[4] + w, ym, xs[5], ym, "IDs or ẑ")
    cv.arrow(xs[5] + w, ym, xs[6], ym, "x̂, ↓2 → 256²", ly=ym - 9)
    cv.arrow(xs[6] + 185, ym, 1305, ym)
    cv.text(1308, ym + 4, "x̂", fs=13, anchor="start", italic=True)
    # S feeds expand / DC (routed above the row)
    cv.line([(xs[1] + 40, y), (xs[1] + 40, 80), (xs[6] + 40, 80), (xs[6] + 40, y)], label="S (maps) also used by expand and DC", lx=430, ly=76)
    # conditioning inputs
    cv.arrow(xs[4] + 45, 30, xs[4] + 45, y, "mask M (column vector)", lx=xs[4] + 45, ly=24)
    cv.arrow(xs[4] + 115, 50, xs[4] + 115, y, "timestep t", lx=xs[4] + 115, ly=44)
    # training branch
    yt = 240
    cv.box(xs[0], yt, w, h, ["fully sampled k-space", "training only"], dashed=True)
    cv.box(xs[1], yt, 180, h, ["oracle maps (ESPIRiT)", "→ SENSE target x₀", "256 × 256, complex"], dashed=True)
    cv.box(xs[3], yt, w, h, ["MedITok encoder", "frozen"], "frozen")
    cv.box(xs[4], yt, w, h, ["training targets", "token IDs → CE loss", "latents z₀ → L2 loss"], dashed=True)
    ymt = yt + h / 2
    cv.arrow(xs[0] + w, ymt, xs[1], ymt)
    cv.arrow(xs[1] + 180, ymt, xs[3], ymt, "x₀, ↑2 → 512²")
    cv.arrow(xs[3] + w, ymt, xs[4], ymt, "z₀ / IDs")
    cv.arrow(xs[4] + 75, yt, xs[4] + 75, y + h, "loss", lx=xs[4] + 90, ly=(yt + y + h) / 2 + 4, anchor="start")
    cv.text(20, 335, "Solid boxes: inference path. Dashed boxes: training only. DC is applied only at inference.", fs=11, anchor="start", italic=True, color="muted")
    return cv


# ---------------------------------------------------------------- Panel B: tokenizer
def panel_b():
    cv = Canvas(1320, 340)
    h = 54
    cv.box(20, 95, 120, h, ["complex image x", "512², ÷ max|x₀|"])
    cv.box(190, 40, 130, 44, ["Re(x)", "grayscale, 3× replicated"])
    cv.box(190, 150, 130, 44, ["Im(x)", "grayscale, 3× replicated"])
    cv.box(370, 95, 150, h, ["ViTamin encoder", "patch 16, shared weights", "both streams"], "frozen")
    cv.box(560, 95, 150, h, ["quant_proj → z", "1024 positions × 64-d", "(8 unit-norm 8-d chunks)"], "frozen")
    cv.box(760, 95, 170, h, ["product quantiser", "chunk k → nearest of 4096", "rows of codebook k"], "frozen")
    cv.box(980, 95, 130, h, ["token IDs", "8 per stream", "per position"])
    cv.arrow(140, 122, 190, 62); cv.arrow(140, 122, 190, 172)
    cv.arrow(320, 62, 370, 110); cv.arrow(320, 172, 370, 136)
    cv.arrow(520, 122, 560, 122); cv.arrow(710, 122, 760, 122, "z"); cv.arrow(930, 122, 980, 122)
    cv.arrow(1110, 122, 1160, 122, "v1 GPT input", lx=1185, ly=118, anchor="start")
    cv.text(1185, 134, "(token lookup)", fs=10.5, anchor="start", color="muted")
    # decoder path
    yd = 230
    cv.box(980, yd, 130, h, ["codebook rows", "idx → f (64-d)"], "frozen")
    cv.box(760, yd, 170, h, ["post_quant_proj →", "ViTamin decoder", "→ grayscale, clamp"], "frozen")
    cv.box(560, yd, 150, h, ["Re, Im decoded", "separately → complex x̂"])
    cv.arrow(1045, 149, 1045, yd)
    cv.arrow(980, yd + 27, 930, yd + 27, "f")
    cv.arrow(760, yd + 27, 710, yd + 27)
    cv.arrow(560, yd + 27, 500, yd + 27, "x̂", lx=520, ly=yd + 22)
    # bypasses
    cv.line([(635, 149), (635, 190), (1180, 190), (1180, 60)], label="v2.1 GPT input: z, quantiser bypassed", lx=900, ly=185, color="muted")
    cv.text(1185, 64, "→ Panel C", fs=10.5, anchor="start", color="muted")
    cv.arrow(845, 316, 845, 284, color="muted")
    cv.text(845, 330, "latent path: ẑ from the latent head enters the decoder here, no quantiser (Panel D)", fs=11, anchor="middle", italic=True, color="muted")
    return cv


# ---------------------------------------------------------------- Panel C: GPT input + body
def panel_c():
    cv = Canvas(1320, 450)
    h = 48
    # v2.1 input
    cv.box(20, 40, 160, h, ["[z_re(p) ; z_im(p)]", "128-d per position p"])
    cv.box(220, 40, 150, h, ["MLP", "128 → 1024 → 1024, GELU"], "new")
    cv.box(410, 40, 90, h, ["RMSNorm"])
    # mask
    cv.box(20, 130, 160, h, ["mask columns m", "±1 per k-space column (256)"])
    cv.box(220, 130, 150, h, ["Linear 256 → 1024"], "new")
    cv.box(410, 130, 130, h, ["RMS-scale × √d", "× m_gate (learned)"], "new")
    # timestep
    cv.box(20, 220, 160, h, ["timestep t", "0 = first pass, ≥1 = refine"])
    cv.box(220, 220, 160, h, ["Embedding (9 × 1024)"])
    cv.box(410, 220, 130, h, ["RMS-scale × √d", "× t_gate, zero at t = 0"], "new")
    # draft (optional)
    cv.box(20, 310, 160, h, ["draft latents", "two-view refinement, optional"], dashed=True)
    cv.box(220, 310, 150, h, ["same MLP"], dashed=True)
    cv.box(410, 310, 140, h, ["RMSNorm_d × d_gate"], dashed=True)
    px, py = 610, 64
    cv.arrow(180, 64, 220, 64); cv.arrow(370, 64, 410, 64); cv.arrow(500, 64, px - 11, py, "e_p")
    for yy, dashed in ((154, False), (244, False), (334, True)):
        cv.arrow(180, yy, 220, yy, dashed=dashed); cv.arrow(370, yy, 410, yy, dashed=dashed)
        cv.line([(540, yy), (575, yy), (575, py + 11)], dashed=dashed, arrow=True)
    cv.plus(px, py)
    cv.text(575, 400, "each conditioning vector is added to every one of the 1024 position tokens", fs=11, italic=True, color="muted")
    # sequence + body
    cv.box(660, 40, 170, h, ["sequence", "[prefix token(t) ; e_1 … e_1024]"])
    cv.box(870, 40, 200, h, ["GPT-L body", "24 × (full attention, 16 heads,", "2-D RoPE; MLP)"], "train")
    cv.box(1110, 40, 130, h, ["hidden state h_p", "1024-d per position"])
    cv.arrow(px + 11, py, 660, 64); cv.arrow(830, 64, 870, 64); cv.arrow(1070, 64, 1110, 64)
    cv.arrow(1175, 88, 1175, 118, "to the heads (Panel D)", lx=1175, ly=132)
    # v1 inset
    cv.region(660, 200, 600, 180, "v1 input, for comparison")
    cv.box(680, 235, 150, h, ["token IDs", "8 Re + 8 Im per position"])
    cv.box(860, 235, 200, h, ["8 Re tables + 8 Im tables", "Σ over codebooks, Σ Re + Im"], "train")
    cv.box(1085, 235, 90, h, ["RMSNorm"])
    cv.arrow(830, 259, 860, 259); cv.arrow(1060, 259, 1085, 259); cv.arrow(1175, 259, 1215, 259, "e_p")
    cv.text(680, 318, "v1: this input, prefix timestep token only (norm ≈ 1 vs ≈ 32 for the tokens: invisible), v1 head.", fs=11, anchor="start", color="muted")
    cv.text(680, 336, "v2: v1 input + mask and timestep conditioning as above.", fs=11, anchor="start", color="muted")
    cv.text(680, 354, "v2.1: continuous input (top left) + mask and timestep conditioning + 8-step head (Panel D).", fs=11, anchor="start", color="muted")
    return cv


# ---------------------------------------------------------------- Panel D: heads and decoding
def panel_d():
    cv = Canvas(1340, 600)
    cv.box(20, 20, 150, 40, ["hidden state h_p", "from the body"])
    h = 44
    # (i) 8-step head
    cv.region(20, 90, 440, 300, "token head, 8 steps (v2.1)")
    cv.box(40, 125, 170, h, ["step 0 input", "h_p + pos₀"])
    cv.box(270, 125, 170, h, ["1-layer transformer", "causal over steps, KV cache"], "train")
    cv.box(270, 195, 170, h, ["RMSNorm → 2 linears", "logits Re_i, Im_i (4096 each)"], "train")
    cv.box(40, 265, 200, 58, ["step i input (i ≥ 1)", "concat E_re(Re_i−1), E_im(Im_i−1)", "Linear 2048→1024 + pos_i"], "new")
    cv.arrow(95, 60, 95, 125, "h_p", lx=110, ly=95, anchor="start")
    cv.arrow(210, 147, 270, 147); cv.arrow(355, 169, 355, 195)
    cv.line([(270, 217), (140, 217), (140, 265)], label="Re_i−1, Im_i−1", lx=200, ly=245)
    cv.line([(240, 294), (255, 294), (255, 160), (270, 160)], arrow=True)
    cv.text(40, 350, "Re_i and Im_i are predicted together at step i.", fs=11, anchor="start", color="muted")
    cv.text(40, 368, "v1 head: one shared table, E(Re) + E(Im) summed (symmetric).", fs=11, anchor="start", color="muted")
    cv.text(40, 386, "init: Linear = [I  I], pos = 0, so it equals the v1 head at the warm start.", fs=11, anchor="start", color="muted")
    # (ii) 16-step head
    cv.region(480, 90, 420, 300, "token head, 16 steps (interleaved, run13)")
    bx = [495, 630, 765]; bw = 125
    cv.box(bx[0], 125, bw, h, ["step 0", "h_p + pos₀ → Re₀"])
    cv.box(bx[1], 125, bw, h, ["step 1", "E_re⁰(Re₀) + pos₁ → Im₀"])
    cv.box(bx[2], 125, bw, h, ["step 2", "E_im⁰(Im₀) + pos₂ → Re₁"])
    cv.arrow(bx[0] + bw, 147, bx[1], 147); cv.arrow(bx[1] + bw, 147, bx[2], 147)
    cv.box(bx[0], 200, bw, h, ["…", "steps 3 to 13"])
    cv.box(bx[1], 200, bw, h, ["step 14", "E_im⁶(Im₆) + pos₁₄ → Re₇"])
    cv.box(bx[2], 200, bw, h, ["step 15", "E_re⁷(Re₇) + pos₁₅ → Im₇"])
    cv.line([(bx[2] + bw / 2, 169), (bx[2] + bw / 2, 185), (bx[0] + bw / 2, 185), (bx[0] + bw / 2, 200)], arrow=True)
    cv.arrow(bx[0] + bw, 222, bx[1], 222); cv.arrow(bx[1] + bw, 222, bx[2], 222)
    cv.text(495, 275, "Same 1-layer transformer, causal over 16 steps; Im_i is conditioned on Re_i.", fs=11, anchor="start", color="muted")
    cv.text(495, 293, "Separate tables per stream and codebook: 8 Re tables, 7 Im tables", fs=11, anchor="start", color="muted")
    cv.text(495, 311, "(Im₇ is never fed back). Learned embedding per chain position.", fs=11, anchor="start", color="muted")
    cv.text(495, 329, "Logits layout identical to the 8-step head: decoding and losses are shared.", fs=11, anchor="start", color="muted")
    # (iii) latent head
    cv.region(920, 90, 400, 300, "latent head (optional, any version)")
    cv.box(940, 125, 150, h, ["RMSNorm", "→ Linear 1024 → 128"], "train")
    cv.box(1120, 125, 180, h, ["split ẑ_re, ẑ_im (64-d)", "unit-norm 8-d chunks"])
    cv.box(1040, 200, 170, h, ["MedITok decoder", "no quantiser (Panel B)"], "frozen")
    cv.arrow(1090, 147, 1120, 147); cv.line([(1210, 169), (1210, 185), (1125, 185), (1125, 200)], arrow=True)
    cv.text(940, 275, "A linear readout of the same hidden state; it predicts the", fs=11, anchor="start", color="muted")
    cv.text(940, 293, "continuous latent directly (the conditional mean), in one pass.", fs=11, anchor="start", color="muted")
    cv.text(940, 311, "Loss: L2 on unit-norm chunks, weight λ = 5, next to the token CE.", fs=11, anchor="start", color="muted")
    cv.text(940, 329, "Trained with the token head; either path is used at inference.", fs=11, anchor="start", color="muted")
    cv.line([(170, 40), (1015, 40), (1015, 125)], arrow=True, label="h_p", lx=600, ly=36)
    # decoding row
    yd = 440
    cv.text(20, 415, "decoding (per position, then the whole image) → per-coil DC", fs=12, anchor="start", bold=True)
    cv.box(20, yd, 200, 54, ["greedy", "argmax IDs → codebook rows", "→ decoder"])
    cv.box(250, yd, 240, 54, ["posterior sampling (PS)", "n sampled draws (top-k) → decoder", "→ mean of the magnitudes"])
    cv.box(520, yd, 240, 54, ["expected embedding", "softmax-weighted codebook rows", "→ decoder, no sampling"])
    cv.box(790, yd, 200, 54, ["latent path", "ẑ → decoder (Panel B)", "one pass"])
    cv.box(1050, yd, 270, 54, ["↓2 to 256², expand S_c · x̂,", "per-coil DC on the measured lines", "(inference only)"], "infer")
    # token logits bus from the two token heads into the three token decodings
    cv.line([(460, 390), (460, 400), (120, 400), (120, yd)], arrow=True)
    cv.line([(370, 400), (370, yd)], arrow=True); cv.line([(640, 400), (640, yd)], arrow=True); cv.line([(460, 400), (640, 400)], arrow=False)
    cv.text(480, 397, "token logits from either token head", fs=11, anchor="start", color="muted")
    cv.line([(1125, 390), (1125, 400), (890, 400), (890, yd)], arrow=True, label="ẑ", lx=1000, ly=396)
    # all four decodings feed DC
    cv.line([(220, yd + 40), (235, yd + 40), (235, yd + 74), (1185, yd + 74), (1185, yd + 54)], arrow=True, color="muted")
    for x in (490, 760, 990):
        cv.line([(x, yd + 40), (x + 12, yd + 40), (x + 12, yd + 74)], arrow=False, color="muted")
    cv.text(20, 545, "Training loss: cross-entropy over the 16 codes of every position (teacher-forced) + λ · L2 of the latent head. No DC in training.", fs=11, anchor="start", italic=True, color="muted")
    cv.text(20, 565, "Results (suno, x8, 256 grid, vs SENSE): v2.1 token head greedy 30.47, PS8 30.98; v2.1 + latent head, latent path 31.27 dB in one pass.", fs=11, anchor="start", italic=True, color="muted")
    return cv


PANELS = [
    ("A", "Reconstruction pipeline", panel_a,
     "Inference runs left to right: the sensitivity net estimates coil maps from the ACS lines of the undersampled k-space, the coils are SENSE-combined into one zero-filled complex image, the image is upsampled to the tokenizer grid and encoded, the GPT predicts the clean representation, the tokenizer decodes it, and the result is expanded to coils and made consistent with the measured k-space lines. Training (dashed) encodes the fully sampled SENSE image with the same tokenizer to obtain token and latent targets; no DC is applied during training."),
    ("B", "Tokenizer (MedITok, SENSE-finetuned, complex pair)", panel_b,
     "The real and imaginary parts are tokenised as two grayscale images by one shared encoder. Each 16×16 patch gives a 64-d latent z split into eight 8-d unit-norm chunks; the product quantiser maps each chunk to one of 4096 codebook rows, giving 8 token IDs per stream per position. v1 feeds the IDs to the GPT; v2.1 feeds z itself. The decoder consumes either codebook rows or a continuous latent, which is what the latent head exploits."),
    ("C", "GPT input and body (v2.1), with the v1 input for comparison", panel_c,
     "In v2.1 the Re and Im latents of each position are concatenated along the channel dimension and projected by an MLP. Three global vectors, all rescaled to the tokens' RMS and multiplied by a learned scalar gate, are added to every position token: the sampling mask (±1 per k-space column), the timestep embedding (zero at timestep 0, so pass 1 of a warm-started model is unchanged), and, in two-view refinement only, the projected latents of the previous reconstruction. The body is the 24-layer GPT-L with full attention and 2-D rotary positions; it emits one hidden state per position."),
    ("D", "Heads and decoding", panel_d,
     "The token heads run one small causal transformer per position over the codebook chain. The 8-step head predicts the Re and Im codes of codebook i together at step i, conditioning on the previous pair through separate per-stream tables, concatenated and projected (initialised to the v1 sum). The 16-step head interleaves Re and Im so that Im_i is conditioned on Re_i. The latent head is a linear readout of the hidden state that predicts the continuous latent and skips the quantiser. Decoding uses greedy IDs, posterior sampling, the expected codebook embedding, or the latent path; all are followed by per-coil DC."),
]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    figs = []
    for key, title, fn, caption in PANELS:
        cv = fn()
        (OUT_DIR / f"arch_{key}_{title.split()[0].lower().strip('(),')}.svg").write_text(cv.render(standalone=True))
        figs.append((key, title, cv.render(standalone=False, label=f"Panel {key}: {title}"), caption))
    css = """
<title>SENSE-GPT Architecture</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap">
<style>
/* layout: one column of figure panels, paper width, captions in serif under each drawing */
:root { --bg:#fbfaf7; --ink:#1f2328; --muted:#6b7075; --train:#dbe7f3; --train_s:#3b6ea5; --frozen:#e9e8e4; --frozen_s:#8a8f94; --new:#fbe9c6; --new_s:#b7791f; --accent:#3b6ea5; --rule:#d9d6cf;
        --f-body:'Source Serif 4', Georgia, serif; --f-ui:'IBM Plex Sans', system-ui, sans-serif; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15171a; --ink:#e6e4df; --muted:#9a9fa5; --train:#233447; --train_s:#7fa6d1; --frozen:#2a2c2f; --frozen_s:#8a8f94; --new:#3d2f14; --new_s:#d9a74a; --accent:#7fa6d1; --rule:#30343a; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#15171a; --ink:#e6e4df; --muted:#9a9fa5; --train:#233447; --train_s:#7fa6d1; --frozen:#2a2c2f; --frozen_s:#8a8f94; --new:#3d2f14; --new_s:#d9a74a; --accent:#7fa6d1; --rule:#30343a; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); padding-block: 28px 48px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1320px; margin: 0 auto; display: grid; gap: 40px; }
header h1 { font-family: var(--f-body); font-weight: 600; font-size: 1.7rem; margin: 0 0 6px; text-wrap: balance; }
header p { margin: 0; max-width: 70ch; font-size: 1rem; line-height: 1.5; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 18px; font-family: var(--f-ui); font-size: 0.82rem; color: var(--muted); margin-top: 14px; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 26px; height: 14px; border-radius: 3px; border: 1.1px solid var(--ink); display: inline-block; }
figure { margin: 0; display: grid; gap: 10px; }
figure .draw { overflow-x: auto; border-top: 1px solid var(--rule); padding-top: 14px; }
figure svg { width: 100%; min-width: 900px; height: auto; display: block; }
figcaption { font-size: 0.95rem; line-height: 1.5; max-width: 95ch; }
figcaption b { font-family: var(--f-ui); font-weight: 600; letter-spacing: 0.2px; }
@media (max-width: 500px) { figure svg { min-width: 820px; } }
</style>
"""
    body = ['<main><header><h1>SENSE-GPT architecture</h1><p>Transformer reconstruction of undersampled multicoil brain MRI: a SENSE-combined zero-filled image is tokenised by a frozen MedITok, a GPT-L predicts the clean representation, and the decoded image is made consistent with the measured k-space. Panels A to D go from the pipeline to the parts of the transformer that differ between model versions.</p>'
            '<div class="legend"><span><i class="sw" style="background:var(--train);border-color:var(--train_s)"></i>trained in this work</span>'
            '<span><i class="sw" style="background:var(--frozen);border-color:var(--frozen_s)"></i>frozen tokenizer</span>'
            '<span><i class="sw" style="background:var(--new);border-color:var(--new_s)"></i>new in v2 / v2.1</span>'
            '<span><i class="sw" style="border-style:dashed"></i>training only, or inference only, as labelled</span></div></header>']
    for key, title, svg, caption in figs:
        body.append(f'<figure><div class="draw">{svg}</div><figcaption><b>Panel {key}. {html.escape(title)}.</b> {html.escape(caption)}</figcaption></figure>')
    body.append("</main>")
    OUT_HTML.write_text(css + "\n".join(body))
    print("wrote", OUT_HTML, "and", sorted(p.name for p in OUT_DIR.glob("arch_*.svg")))


if __name__ == "__main__":
    main()
