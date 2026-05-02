"""
====================================================================
EDA – Bagian 1: Karakteristik Demografis
Dataset Pasien Sirosis Hati (n=627, 137 fitur)
====================================================================
Variabel yang dianalisis:
  • Usia
  • Jenis Kelamin (jk)
  • IMT & Kategori IMT
  • Daerah Asal
  • Etiologi
  • Riwayat Alkohol
  • CTP Score
  • Durasi Penyakit (kualitatif)
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

# ─── 0. KONFIGURASI VISUAL ───────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#2a3444",
    "axes.labelcolor":   "#8b949e",
    "axes.titlecolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#2a3444",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Palet warna utama
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
GOLD   = "#d29922"
RED    = "#f78166"
PURPLE = "#bc8cff"
MUTED  = "#8b949e"
DIM    = "#6e7681"
TEXT   = "#e6edf3"

PALETTE = [BLUE, GREEN, GOLD, RED, PURPLE, "#39d353", "#ffa657"]

# ─── 1. LOAD DATA ────────────────────────────────────────────────
FILE = "data_final_clean_manual.xlsx"   # sesuaikan path jika perlu

print("=" * 60)
print("MEMUAT DATASET ...")
print("=" * 60)

df = pd.read_excel(FILE, sheet_name="Data")
N = len(df)
print(f"  Total baris  : {N}")
print(f"  Total kolom  : {len(df.columns)}")

# ─── 2. PREPROCESSING DEMOGRAFIS ─────────────────────────────────
print("\n" + "=" * 60)
print("PRE-PROCESSING DEMOGRAFIS")
print("=" * 60)

# --- Usia: buang outlier jelas (usia=0 kemungkinan error entry) ---
usia_raw = df["usia"].copy()
df.loc[df["usia"] == 0, "usia"] = np.nan      # P0579, usia=0 → missing
print(f"  Usia: {usia_raw.isna().sum()} missing awal → "
      f"{df['usia'].isna().sum()} setelah koreksi usia=0")

# --- IMT: buang outlier ekstrem (IMT=200, kemungkinan entry error) ---
imt_raw = df["imt"].copy()
df.loc[df["imt"] > 100, "imt"] = np.nan       # P0291, IMT=200 → missing
print(f"  IMT : {imt_raw.isna().sum()} missing awal → "
      f"{df['imt'].isna().sum()} setelah koreksi IMT>100")

# --- Kelompok usia ---
bins   = [0, 30, 40, 50, 60, 70, 80, 101]
labels = ["<30", "30–39", "40–49", "50–59", "60–69", "70–79", "≥80"]
df["age_group"] = pd.cut(df["usia"], bins=bins, labels=labels, right=False)

# --- Ringkasan missing per kolom demografis ---
demo_cols = {
    "usia":         "Usia (tahun)",
    "jk":           "Jenis Kelamin",
    "imt":          "IMT (kg/m²)",
    "imt_kategori": "Kategori IMT",
    "daerah_asal":  "Daerah Asal",
    "etiologi":     "Etiologi",
    "alkohol":      "Riwayat Alkohol",
    "ctp":          "Skor CTP",
    "durasi_penyakit": "Durasi Penyakit",
}

print(f"\n  {'Kolom':<20} {'Missing':>8} {'%':>7}")
print(f"  {'-'*38}")
for col, label in demo_cols.items():
    miss = df[col].isna().sum()
    pct  = miss / N * 100
    print(f"  {label:<20} {miss:>8}  {pct:>5.1f}%")


# ─── 3. STATISTIK DESKRIPTIF ─────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTIK DESKRIPTIF")
print("=" * 60)

# ── 3a. Usia ──
usia = df["usia"].dropna()
print(f"\n[Usia] n={len(usia)}, missing={df['usia'].isna().sum()}")
print(f"  Mean ± SD : {usia.mean():.1f} ± {usia.std():.1f} tahun")
print(f"  Median    : {usia.median():.0f} tahun")
print(f"  IQR       : [{usia.quantile(.25):.0f}, {usia.quantile(.75):.0f}]")
print(f"  Min–Max   : {usia.min():.0f} – {usia.max():.0f}")
print(f"\n  Kelompok Usia:")
for grp, cnt in df["age_group"].value_counts().sort_index().items():
    pct = cnt / N * 100
    print(f"    {grp:<8} : {cnt:>4}  ({pct:.1f}%)")

# ── 3b. Jenis kelamin ──
jk_counts = df["jk"].value_counts()
print(f"\n[Jenis Kelamin] n={jk_counts.sum()}, missing={df['jk'].isna().sum()}")
for k, v in jk_counts.items():
    label = "Laki-laki" if k == "L" else "Perempuan"
    print(f"  {label:<12}: {v:>4}  ({v/jk_counts.sum()*100:.1f}%)")

# ── 3c. IMT ──
imt = df["imt"].dropna()
print(f"\n[IMT] n={len(imt)}, missing={df['imt'].isna().sum()} "
      f"({df['imt'].isna().sum()/N*100:.1f}%)")
print(f"  Mean ± SD : {imt.mean():.1f} ± {imt.std():.1f}")
print(f"  Median    : {imt.median():.1f}")
print(f"  IQR       : [{imt.quantile(.25):.1f}, {imt.quantile(.75):.1f}]")
print(f"\n  Kategori IMT:")
imt_cat = df["imt_kategori"].value_counts()
order_imt = ["underweight", "normal", "overweight", "obese"]
for cat in order_imt:
    if cat in imt_cat.index:
        v = imt_cat[cat]
        print(f"    {cat:<15}: {v:>4}  ({v/imt_cat.sum()*100:.1f}%)")

# ── 3d. Etiologi ──
etio = df["etiologi"].dropna().value_counts()
print(f"\n[Etiologi] n={etio.sum()}, missing={df['etiologi'].isna().sum()} "
      f"({df['etiologi'].isna().sum()/N*100:.1f}%)")
for k, v in etio.items():
    print(f"  {k:<25}: {v:>4}  ({v/etio.sum()*100:.1f}%)")

# ── 3e. Alkohol ──
alk = df["alkohol"].dropna()
n_pos = int(alk.sum())
n_neg = len(alk) - n_pos
print(f"\n[Alkohol] n={len(alk)}, missing={df['alkohol'].isna().sum()} "
      f"({df['alkohol'].isna().sum()/N*100:.1f}%)")
print(f"  Positif   : {n_pos:>4}  ({n_pos/len(alk)*100:.1f}%)")
print(f"  Negatif   : {n_neg:>4}  ({n_neg/len(alk)*100:.1f}%)")

# ── 3f. CTP ──
ctp = df["ctp"].dropna().value_counts()
print(f"\n[Skor CTP] n={ctp.sum()}, missing={df['ctp'].isna().sum()} "
      f"({df['ctp'].isna().sum()/N*100:.1f}%)")
for grp in ["A", "B", "C"]:
    if grp in ctp.index:
        v = ctp[grp]
        print(f"  CTP-{grp}    : {v:>4}  ({v/ctp.sum()*100:.1f}%)")

# ── 3g. Daerah asal ──
daerah = df["daerah_asal"].dropna().value_counts()
print(f"\n[Daerah Asal] n={daerah.sum()}, missing={df['daerah_asal'].isna().sum()}")
for k, v in daerah.items():
    print(f"  {k:<20}: {v:>4}  ({v/daerah.sum()*100:.1f}%)")


# ─── 4. VISUALISASI ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEMBUAT VISUALISASI ...")
print("=" * 60)

def add_value_labels(ax, bars, fmt="{:.0f}", fontsize=8.5, color=TEXT, offset=2):
    """Tambahkan label nilai di atas setiap bar."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="600"
        )

def style_ax(ax, title, xlabel="", ylabel="", grid_axis="y"):
    ax.set_title(title, fontsize=11, fontweight="700", color=TEXT, pad=12)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED, labelpad=6)
    if grid_axis:
        ax.grid(axis=grid_axis, linewidth=0.6, color="#2a3444", alpha=0.8)
        ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3444")
    ax.tick_params(colors=MUTED, labelsize=8.5)


# ════════════════════════════════════════════════════════════════
# FIGURE 1 – Usia: histogram + boxplot + kelompok usia
# ════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.patch.set_facecolor("#0d1117")
fig1.suptitle("Distribusi Usia Pasien", fontsize=14, fontweight="800",
              color=TEXT, y=1.01)

usia_valid = df["usia"].dropna()

# — Histogram —
ax = axes[0]
n_bins = 18
counts, edges, patches = ax.hist(usia_valid, bins=n_bins, color=BLUE,
                                  edgecolor="#0d1117", linewidth=0.5, alpha=0.85)
# Gradient warna berdasarkan frekuensi
max_count = counts.max()
cmap = plt.cm.Blues
for patch, c in zip(patches, counts):
    patch.set_facecolor(cmap(0.3 + 0.7 * c / max_count))

ax.axvline(usia_valid.mean(),   color=GOLD,  lw=1.8, ls="--", label=f"Mean  {usia_valid.mean():.1f}")
ax.axvline(usia_valid.median(), color=GREEN, lw=1.8, ls="-",  label=f"Median {usia_valid.median():.0f}")
ax.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#2a3444", labelcolor=TEXT)
style_ax(ax, "Distribusi Usia (Histogram)", "Usia (tahun)", "Frekuensi")

# — Boxplot —
ax = axes[1]
bp = ax.boxplot(usia_valid, vert=True, patch_artist=True, widths=0.5,
                medianprops=dict(color=GREEN, lw=2),
                boxprops=dict(facecolor=BLUE, alpha=0.4, edgecolor=BLUE),
                whiskerprops=dict(color=MUTED, lw=1.2),
                capprops=dict(color=MUTED, lw=1.2),
                flierprops=dict(marker="o", markerfacecolor=RED, markersize=4,
                                markeredgecolor=RED, alpha=0.6))
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#2a3444")
ax.grid(axis="y", linewidth=0.6, color="#2a3444", alpha=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)

# Annotasi statistik
stats_text = (f"n     = {len(usia_valid)}\n"
              f"Mean  = {usia_valid.mean():.1f}\n"
              f"SD    = {usia_valid.std():.1f}\n"
              f"Median= {usia_valid.median():.0f}\n"
              f"Q1    = {usia_valid.quantile(.25):.0f}\n"
              f"Q3    = {usia_valid.quantile(.75):.0f}\n"
              f"Min   = {usia_valid.min():.0f}\n"
              f"Max   = {usia_valid.max():.0f}")
ax.text(1.35, usia_valid.median(), stats_text, fontsize=8.5,
        color=MUTED, va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1c2230",
                  edgecolor="#2a3444", alpha=0.9),
        fontfamily="monospace")
ax.set_title("Boxplot Usia", fontsize=11, fontweight="700", color=TEXT, pad=12)
ax.set_ylabel("Usia (tahun)", fontsize=9, color=MUTED)

# — Bar chart kelompok usia —
ax = axes[2]
grp_counts = df["age_group"].value_counts().sort_index()
colors_grp = [BLUE if g not in ["50–59", "60–69"] else GOLD for g in grp_counts.index]
bars = ax.bar(grp_counts.index, grp_counts.values,
              color=colors_grp, edgecolor="#0d1117", linewidth=0.5, alpha=0.88)
add_value_labels(ax, bars, offset=0.8)
style_ax(ax, "Kelompok Usia", "Kelompok Usia", "Jumlah Pasien")
ax.set_ylim(0, grp_counts.max() * 1.18)

fig1.tight_layout()
fig1.savefig("fig1_usia.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig1_usia.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 2 – Jenis Kelamin
# ════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(11, 5))
fig2.patch.set_facecolor("#0d1117")
fig2.suptitle("Distribusi Jenis Kelamin", fontsize=14,
              fontweight="800", color=TEXT, y=1.01)

jk_valid = df["jk"].dropna()
jk_cnt   = jk_valid.value_counts()  # L, P
jk_labels = ["Laki-laki", "Perempuan"]
jk_vals   = [jk_cnt.get("L", 0), jk_cnt.get("P", 0)]
jk_colors = [BLUE, "#f0aab0"]

# — Pie chart —
ax = axes[0]
wedges, texts, autotexts = ax.pie(
    jk_vals, labels=jk_labels, colors=jk_colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
    textprops=dict(color=TEXT, fontsize=10))
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("700")
    at.set_color("#0d1117")
ax.set_title("Proporsi Jenis Kelamin\n(non-missing)", fontsize=11,
             fontweight="700", color=TEXT, pad=10)

# — Bar chart dengan breakdown per kelompok usia —
ax = axes[1]
age_jk = df.dropna(subset=["age_group", "jk"]).groupby(
    ["age_group", "jk"], observed=True).size().unstack(fill_value=0)
x = np.arange(len(age_jk.index))
w = 0.38
b1 = ax.bar(x - w/2, age_jk.get("L", 0), width=w, color=BLUE,
            edgecolor="#0d1117", lw=0.5, alpha=0.88, label="Laki-laki")
b2 = ax.bar(x + w/2, age_jk.get("P", 0), width=w, color="#f0aab0",
            edgecolor="#0d1117", lw=0.5, alpha=0.88, label="Perempuan")
ax.set_xticks(x)
ax.set_xticklabels(age_jk.index, rotation=0, fontsize=8.5)
ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#2a3444", labelcolor=TEXT)
style_ax(ax, "Jenis Kelamin per Kelompok Usia", "Kelompok Usia", "Jumlah Pasien")
ax.set_ylim(0, max(age_jk.max()) * 1.2)
add_value_labels(ax, b1, offset=0.5, fontsize=7.5)
add_value_labels(ax, b2, offset=0.5, fontsize=7.5)

fig2.tight_layout()
fig2.savefig("fig2_jk.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig2_jk.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 3 – IMT
# ════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
fig3.patch.set_facecolor("#0d1117")
fig3.suptitle("Indeks Massa Tubuh (IMT)", fontsize=14,
              fontweight="800", color=TEXT, y=1.01)

imt_valid = df["imt"].dropna()

# — Histogram IMT —
ax = axes[0]
_, edges2, patches2 = ax.hist(imt_valid, bins=20, color=GREEN,
                               edgecolor="#0d1117", lw=0.5, alpha=0.85)
# Zona warna IMT
ax.axvspan(imt_valid.min(), 18.5,  alpha=0.12, color=BLUE,   label="Underweight (<18.5)")
ax.axvspan(18.5, 22.9,             alpha=0.12, color=GREEN,  label="Normal (18.5–22.9)")
ax.axvspan(22.9, 24.9,             alpha=0.12, color=GOLD,   label="Overweight (23–24.9)")
ax.axvspan(24.9, imt_valid.max(),  alpha=0.12, color=RED,    label="Obese (≥25)")
ax.axvline(imt_valid.median(), color=GOLD, lw=1.8, ls="--",
           label=f"Median {imt_valid.median():.1f}")
ax.legend(fontsize=7.5, facecolor="#161b22", edgecolor="#2a3444",
          labelcolor=TEXT, loc="upper right")
style_ax(ax, "Distribusi IMT (Histogram)", "IMT (kg/m²)", "Frekuensi")
ax.set_xlim(12, imt_valid.max() + 1)

# — Pie kategori IMT —
ax = axes[1]
order_c = ["underweight", "normal", "overweight", "obese"]
label_c = ["Underweight", "Normal", "Overweight", "Obese"]
color_c = [BLUE, GREEN, GOLD, RED]
cat_df  = df["imt_kategori"].dropna()
vals_c  = [cat_df.value_counts().get(c, 0) for c in order_c]
wedges2, texts2, auto2 = ax.pie(
    vals_c, labels=label_c, colors=color_c,
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
    textprops=dict(color=TEXT, fontsize=9.5))
for at in auto2:
    at.set_fontsize(10)
    at.set_fontweight("700")
    at.set_color("#0d1117")
ax.set_title(f"Kategori IMT (n={int(cat_df.count())})\nmissing={df['imt'].isna().sum()}",
             fontsize=11, fontweight="700", color=TEXT, pad=10)

# — IMT per jenis kelamin (boxplot) —
ax = axes[2]
imt_l = df[df["jk"] == "L"]["imt"].dropna()
imt_p = df[df["jk"] == "P"]["imt"].dropna()
bp3 = ax.boxplot([imt_l, imt_p], vert=True, patch_artist=True,
                  widths=0.45, labels=["Laki-laki", "Perempuan"],
                  medianprops=dict(color=GOLD, lw=2),
                  boxprops=dict(alpha=0.5),
                  whiskerprops=dict(lw=1.2, color=MUTED),
                  capprops=dict(lw=1.2, color=MUTED),
                  flierprops=dict(marker="o", markersize=3.5,
                                  markeredgecolor=RED, alpha=0.5))
bp3["boxes"][0].set_facecolor(BLUE)
bp3["boxes"][1].set_facecolor("#f0aab0")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a3444")
ax.grid(axis="y", color="#2a3444", lw=0.6, alpha=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_title("IMT per Jenis Kelamin", fontsize=11, fontweight="700",
             color=TEXT, pad=12)
ax.set_ylabel("IMT (kg/m²)", fontsize=9, color=MUTED)
# Tambahkan n
for i, (label, data) in enumerate(zip(["L", "P"], [imt_l, imt_p]), start=1):
    ax.text(i, ax.get_ylim()[0] - 0.8, f"n={len(data)}",
            ha="center", fontsize=8.5, color=MUTED)

fig3.tight_layout()
fig3.savefig("fig3_imt.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig3_imt.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 4 – Etiologi & Alkohol
# ════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 3, figsize=(16, 5))
fig4.patch.set_facecolor("#0d1117")
fig4.suptitle("Etiologi Sirosis & Riwayat Alkohol", fontsize=14,
              fontweight="800", color=TEXT, y=1.01)

# — Pie etiologi —
ax = axes[0]
etio_cnt = df["etiologi"].dropna().value_counts()
etio_c   = [GOLD, BLUE, RED]
wedges3, texts3, auto3 = ax.pie(
    etio_cnt.values, labels=etio_cnt.index, colors=etio_c,
    autopct="%1.1f%%", startangle=120,
    wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
    textprops=dict(color=TEXT, fontsize=9))
for at in auto3:
    at.set_fontsize(10.5)
    at.set_fontweight("700")
    at.set_color("#0d1117")
ax.set_title(f"Distribusi Etiologi (n={etio_cnt.sum()})\n"
             f"missing={df['etiologi'].isna().sum()} ({df['etiologi'].isna().sum()/N*100:.0f}%)",
             fontsize=11, fontweight="700", color=TEXT, pad=10)

# — Bar etiologi detail —
ax = axes[1]
bars4 = ax.barh(etio_cnt.index[::-1], etio_cnt.values[::-1],
                color=[GOLD, BLUE, RED][::-1],
                edgecolor="#0d1117", lw=0.5, alpha=0.88)
for bar, val in zip(bars4, etio_cnt.values[::-1]):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
            f"{val} ({val/etio_cnt.sum()*100:.1f}%)",
            va="center", fontsize=9, color=TEXT, fontweight="600")
style_ax(ax, "Jumlah Pasien per Etiologi", "Jumlah Pasien", "", grid_axis="x")
ax.set_xlim(0, etio_cnt.max() * 1.3)
ax.grid(axis="x", color="#2a3444", lw=0.6, alpha=0.8)
ax.set_axisbelow(True)

# — Alkohol donut chart —
ax = axes[2]
alk_valid = df["alkohol"].dropna()
alk_pos   = int(alk_valid.sum())
alk_neg   = len(alk_valid) - alk_pos
alk_miss  = df["alkohol"].isna().sum()
alk_vals2 = [alk_neg, alk_pos, alk_miss]
alk_lab2  = [f"Negatif\n({alk_neg})", f"Positif\n({alk_pos})",
             f"Missing\n({alk_miss})"]
alk_col2  = [GREEN, RED, DIM]

wedge_kw = dict(width=0.42, edgecolor="#0d1117", linewidth=2)
ax.pie(alk_vals2, labels=alk_lab2, colors=alk_col2,
       autopct="%1.1f%%", startangle=90,
       wedgeprops=wedge_kw,
       textprops=dict(color=TEXT, fontsize=9.5))
ax.set_title("Riwayat Konsumsi Alkohol\n(termasuk data missing)",
             fontsize=11, fontweight="700", color=TEXT, pad=10)

fig4.tight_layout()
fig4.savefig("fig4_etiologi_alkohol.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig4_etiologi_alkohol.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 5 – Daerah Asal
# ════════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 2, figsize=(14, 5))
fig5.patch.set_facecolor("#0d1117")
fig5.suptitle("Distribusi Daerah Asal Pasien", fontsize=14,
              fontweight="800", color=TEXT, y=1.01)

daerah_cnt = df["daerah_asal"].dropna().value_counts()

# — Horizontal bar —
ax = axes[0]
colors_d = [BLUE if i == 0 else (GOLD if i == 1 else MUTED)
            for i in range(len(daerah_cnt))]
bars5 = ax.barh(daerah_cnt.index[::-1], daerah_cnt.values[::-1],
                color=colors_d[::-1], edgecolor="#0d1117", lw=0.5, alpha=0.88)
for bar, val in zip(bars5, daerah_cnt.values[::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f"{val} ({val/daerah_cnt.sum()*100:.1f}%)",
            va="center", fontsize=8.5, color=TEXT, fontweight="600")
style_ax(ax, "Daerah Asal (Semua)", "Jumlah Pasien", "", grid_axis="x")
ax.set_xlim(0, daerah_cnt.max() * 1.25)
ax.grid(axis="x", color="#2a3444", lw=0.6, alpha=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=9)

# — Donut: Minang vs non-Minang —
ax = axes[1]
n_minang    = daerah_cnt.get("Minang", 0)
n_non_minang = daerah_cnt.sum() - n_minang
n_miss_d     = df["daerah_asal"].isna().sum()
wedge_kw2 = dict(width=0.42, edgecolor="#0d1117", linewidth=2)
ax.pie(
    [n_minang, n_non_minang, n_miss_d],
    labels=[f"Minang\n({n_minang})",
            f"Non-Minang\n({n_non_minang})",
            f"Missing\n({n_miss_d})"],
    colors=[BLUE, PURPLE, DIM],
    autopct="%1.1f%%", startangle=80,
    wedgeprops=wedge_kw2,
    textprops=dict(color=TEXT, fontsize=9.5)
)
ax.set_title("Minang vs Non-Minang\n(inkl. missing)",
             fontsize=11, fontweight="700", color=TEXT, pad=10)

fig5.tight_layout()
fig5.savefig("fig5_daerah_asal.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig5_daerah_asal.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 6 – CTP Score
# ════════════════════════════════════════════════════════════════
fig6, axes = plt.subplots(1, 3, figsize=(16, 5))
fig6.patch.set_facecolor("#0d1117")
fig6.suptitle("Distribusi Skor Child-Turcotte-Pugh (CTP)", fontsize=14,
              fontweight="800", color=TEXT, y=1.01)

ctp_valid = df["ctp"].dropna()
ctp_cnt   = ctp_valid.value_counts().reindex(["A", "B", "C"])
ctp_colors = [GREEN, GOLD, RED]
ctp_labels = ["CTP-A\n(Ringan)", "CTP-B\n(Sedang)", "CTP-C\n(Berat)"]

# — Bar chart CTP —
ax = axes[0]
bars6 = ax.bar(ctp_labels, ctp_cnt.values,
               color=ctp_colors, edgecolor="#0d1117", lw=0.5,
               alpha=0.88, width=0.55)
add_value_labels(ax, bars6, offset=1)
for bar, val in zip(bars6, ctp_cnt.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + ctp_cnt.max() * 0.06,
            f"({val/ctp_cnt.sum()*100:.1f}%)",
            ha="center", fontsize=8.5, color=MUTED)
style_ax(ax, f"Distribusi CTP (n={ctp_cnt.sum()})\nmissing={df['ctp'].isna().sum()}",
         "", "Jumlah Pasien")
ax.set_ylim(0, ctp_cnt.max() * 1.22)

# — Pie CTP —
ax = axes[1]
wedges6, texts6, auto6 = ax.pie(
    ctp_cnt.values, labels=["CTP-A", "CTP-B", "CTP-C"],
    colors=ctp_colors, autopct="%1.1f%%", startangle=80,
    wedgeprops=dict(edgecolor="#0d1117", linewidth=2.5),
    textprops=dict(color=TEXT, fontsize=10.5))
for at in auto6:
    at.set_fontsize(11.5)
    at.set_fontweight("700")
    at.set_color("#0d1117")
ax.set_title("Proporsi CTP", fontsize=11, fontweight="700", color=TEXT, pad=10)

# — CTP per jenis kelamin (stacked bar) —
ax = axes[2]
ctp_jk = df.dropna(subset=["ctp", "jk"]).groupby(
    ["ctp", "jk"]).size().unstack(fill_value=0).reindex(["A", "B", "C"])
x6     = np.arange(3)
bottom = np.zeros(3)
jk_colors6 = [BLUE, "#f0aab0"]
jk_labels6 = ["Laki-laki", "Perempuan"]
for jk_key, color, label in zip(["L", "P"], jk_colors6, jk_labels6):
    vals6 = ctp_jk.get(jk_key, pd.Series([0]*3)).values
    ax.bar(x6, vals6, bottom=bottom, color=color,
           label=label, edgecolor="#0d1117", lw=0.5, alpha=0.88, width=0.55)
    # Label di tengah segmen
    for xi, (v, b) in enumerate(zip(vals6, bottom)):
        if v > 5:
            ax.text(xi, b + v/2, str(v),
                    ha="center", va="center", fontsize=9,
                    color="#0d1117", fontweight="700")
    bottom += vals6
ax.set_xticks(x6)
ax.set_xticklabels(ctp_labels, fontsize=9)
ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#2a3444", labelcolor=TEXT)
style_ax(ax, "CTP per Jenis Kelamin (Stacked)", "", "Jumlah Pasien")

fig6.tight_layout()
fig6.savefig("fig6_ctp.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig6_ctp.png")


# ════════════════════════════════════════════════════════════════
# FIGURE 7 – Dashboard Ringkasan Demografis
# ════════════════════════════════════════════════════════════════
fig7 = plt.figure(figsize=(18, 10))
fig7.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(2, 4, figure=fig7, hspace=0.45, wspace=0.4)

# Judul utama
fig7.text(0.5, 0.97, "Dashboard Demografis — Pasien Sirosis Hati",
          ha="center", va="top", fontsize=15, fontweight="800",
          color=TEXT)
fig7.text(0.5, 0.94, f"n = {N} pasien  |  137 fitur  |  Dataset: April 2026",
          ha="center", va="top", fontsize=10, color=MUTED)

# [0,0] Kelompok usia
ax70 = fig7.add_subplot(gs[0, 0])
grp_c = df["age_group"].value_counts().sort_index()
bar_c = [GREEN if g in ["50–59", "60–69"] else BLUE for g in grp_c.index]
ax70.bar(grp_c.index, grp_c.values, color=bar_c,
         edgecolor="#0d1117", lw=0.5, alpha=0.88)
style_ax(ax70, "Kelompok Usia", "Usia", "n", grid_axis="y")
ax70.tick_params(axis="x", labelsize=8, rotation=30)
ax70.set_ylim(0, grp_c.max() * 1.2)
for xi, (label, v) in enumerate(zip(grp_c.index, grp_c.values)):
    ax70.text(xi, v + 1, str(v), ha="center", fontsize=8, color=MUTED)

# [0,1] Jenis kelamin
ax71 = fig7.add_subplot(gs[0, 1])
jk_v = [jk_cnt.get("L", 0), jk_cnt.get("P", 0)]
wedge_kw3 = dict(edgecolor="#0d1117", linewidth=2)
ax71.pie(jk_v, labels=["L", "P"], colors=[BLUE, "#f0aab0"],
         autopct="%1.1f%%", startangle=90,
         wedgeprops=wedge_kw3,
         textprops=dict(color=TEXT, fontsize=10))
ax71.set_title(f"Jenis Kelamin\n(L={jk_v[0]}, P={jk_v[1]})",
               fontsize=10, fontweight="700", color=TEXT, pad=8)

# [0,2] CTP
ax72 = fig7.add_subplot(gs[0, 2])
ax72.pie(ctp_cnt.values, labels=[f"A\n{ctp_cnt['A']}", f"B\n{ctp_cnt['B']}", f"C\n{ctp_cnt['C']}"],
         colors=ctp_colors, autopct="%1.0f%%", startangle=80,
         wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
         textprops=dict(color=TEXT, fontsize=9.5))
ax72.set_title(f"Skor CTP (n={ctp_cnt.sum()})", fontsize=10,
               fontweight="700", color=TEXT, pad=8)

# [0,3] Etiologi
ax73 = fig7.add_subplot(gs[0, 3])
ax73.pie(etio_cnt.values, labels=etio_cnt.index, colors=[GOLD, BLUE, RED],
         autopct="%1.1f%%", startangle=120,
         wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
         textprops=dict(color=TEXT, fontsize=9))
ax73.set_title(f"Etiologi (n={etio_cnt.sum()})", fontsize=10,
               fontweight="700", color=TEXT, pad=8)

# [1,0] IMT kategori
ax74 = fig7.add_subplot(gs[1, 0])
imt_o = [imt_cat.get(c, 0) for c in order_imt]
bars7 = ax74.bar(["Underw.", "Normal", "Overw.", "Obese"],
                 imt_o, color=[BLUE, GREEN, GOLD, RED],
                 edgecolor="#0d1117", lw=0.5, alpha=0.88)
style_ax(ax74, f"Kategori IMT (n={sum(imt_o)})", "", "n")
for bar, v in zip(bars7, imt_o):
    ax74.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              str(v), ha="center", fontsize=8.5, color=MUTED)
ax74.set_ylim(0, max(imt_o) * 1.2)

# [1,1] Daerah asal top-6
ax75 = fig7.add_subplot(gs[1, 1])
daerah_top6 = daerah_cnt.head(6)
col_d = [BLUE if i == 0 else (GOLD if i == 1 else MUTED)
         for i in range(len(daerah_top6))]
ax75.barh(daerah_top6.index[::-1], daerah_top6.values[::-1],
          color=col_d[::-1], edgecolor="#0d1117", lw=0.5, alpha=0.88)
for v, y in zip(daerah_top6.values[::-1], range(len(daerah_top6))):
    ax75.text(v + 2, y, str(v), va="center", fontsize=8.5, color=TEXT)
style_ax(ax75, "Top-6 Daerah Asal", "n", "", grid_axis="x")
ax75.set_xlim(0, daerah_top6.max() * 1.2)
ax75.grid(axis="x", color="#2a3444", lw=0.6, alpha=0.8)
ax75.tick_params(axis="y", labelsize=9)

# [1,2] Histogram usia ringkas
ax76 = fig7.add_subplot(gs[1, 2])
ax76.hist(usia_valid, bins=16, color=BLUE, edgecolor="#0d1117",
          lw=0.5, alpha=0.85)
ax76.axvline(usia_valid.mean(),   color=GOLD,  lw=1.8, ls="--",
             label=f"Mean {usia_valid.mean():.1f}th")
ax76.axvline(usia_valid.median(), color=GREEN, lw=1.8, ls="-",
             label=f"Median {usia_valid.median():.0f}th")
ax76.legend(fontsize=8, facecolor="#161b22", edgecolor="#2a3444",
            labelcolor=TEXT)
style_ax(ax76, "Distribusi Usia", "Usia (tahun)", "n")

# [1,3] Summary stat card (text)
ax77 = fig7.add_subplot(gs[1, 3])
ax77.set_facecolor("#0d1117")
for spine in ax77.spines.values():
    spine.set_visible(False)
ax77.set_xticks([]); ax77.set_yticks([])
summary_lines = [
    ("STATISTIK RINGKAS", TEXT,  11, True),
    ("", MUTED, 9, False),
    (f"Usia: {usia_valid.mean():.1f}±{usia_valid.std():.1f} thn",  TEXT, 9, False),
    (f"Median usia: {usia_valid.median():.0f} thn",             MUTED, 9, False),
    (f"Range: {usia_valid.min():.0f}–{usia_valid.max():.0f} thn", MUTED, 9, False),
    ("", MUTED, 9, False),
    (f"IMT: {imt_valid.mean():.1f}±{imt_valid.std():.1f} kg/m²", TEXT, 9, False),
    (f"Median IMT: {imt_valid.median():.1f} kg/m²",            MUTED, 9, False),
    ("", MUTED, 9, False),
    (f"CTP-C (berat): {ctp_cnt['C']} ({ctp_cnt['C']/ctp_cnt.sum()*100:.1f}%)", RED, 9, True),
    (f"CTP-B (sedang): {ctp_cnt['B']} ({ctp_cnt['B']/ctp_cnt.sum()*100:.1f}%)", GOLD, 9, False),
    (f"CTP-A (ringan): {ctp_cnt['A']} ({ctp_cnt['A']/ctp_cnt.sum()*100:.1f}%)", GREEN, 9, False),
    ("", MUTED, 9, False),
    (f"Alkohol (+): {n_pos} pasien ({n_pos/len(alk_valid)*100:.1f}%)", MUTED, 9, False),
]
y_pos = 0.96
for text, color, size, bold in summary_lines:
    ax77.text(0.05, y_pos, text, transform=ax77.transAxes,
              fontsize=size, color=color,
              fontweight="700" if bold else "400",
              va="top", fontfamily="monospace")
    y_pos -= 0.07

fig7.savefig("fig7_dashboard_demografi.png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
print("  ✓ fig7_dashboard_demografi.png")


# ─── 5. RINGKASAN AKHIR ──────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN ANALISIS DEMOGRAFIS")
print("=" * 60)
print(f"""
  Total pasien          : {N}
  Usia rata-rata        : {usia_valid.mean():.1f} ± {usia_valid.std():.1f} tahun (median {usia_valid.median():.0f})
  Rentang usia          : {usia_valid.min():.0f} – {usia_valid.max():.0f} tahun
  Kelompok terbanyak    : 50–59 tahun ({df['age_group'].value_counts()['50–59']} pasien)

  Laki-laki             : {jk_cnt.get('L',0)} ({jk_cnt.get('L',0)/jk_cnt.sum()*100:.1f}%)
  Perempuan             : {jk_cnt.get('P',0)} ({jk_cnt.get('P',0)/jk_cnt.sum()*100:.1f}%)

  CTP-A                 : {ctp_cnt['A']} ({ctp_cnt['A']/ctp_cnt.sum()*100:.1f}%)
  CTP-B                 : {ctp_cnt['B']} ({ctp_cnt['B']/ctp_cnt.sum()*100:.1f}%)
  CTP-C                 : {ctp_cnt['C']} ({ctp_cnt['C']/ctp_cnt.sum()*100:.1f}%)

  Etiologi HepB         : {etio_cnt.get('Hepatitis B',0)} ({etio_cnt.get('Hepatitis B',0)/etio_cnt.sum()*100:.1f}%)
  Etiologi HepC         : {etio_cnt.get('Hepatitis C',0)} ({etio_cnt.get('Hepatitis C',0)/etio_cnt.sum()*100:.1f}%)
  Etiologi Tdk Diket.   : {etio_cnt.get('Tidak diketahui',0)} ({etio_cnt.get('Tidak diketahui',0)/etio_cnt.sum()*100:.1f}%)

  Alkohol positif       : {n_pos} ({n_pos/len(alk_valid)*100:.1f}% dari yang tercatat)
  Asal daerah Minang    : {n_minang} ({n_minang/daerah_cnt.sum()*100:.1f}%)

  Catatan data quality  :
    - Usia missing      : {df['usia'].isna().sum()} (setelah koreksi usia=0 pada P0579)
    - IMT missing       : {df['imt'].isna().sum()} ({df['imt'].isna().sum()/N*100:.1f}%)
    - Etiologi missing  : {df['etiologi'].isna().sum()} ({df['etiologi'].isna().sum()/N*100:.1f}%)
    - Alkohol missing   : {df['alkohol'].isna().sum()} ({df['alkohol'].isna().sum()/N*100:.1f}%)
    - CTP missing       : {df['ctp'].isna().sum()} ({df['ctp'].isna().sum()/N*100:.1f}%)
    - IMT outlier P0291 (IMT=200) → dikecualikan dari analisis IMT
""")

print("=" * 60)
print("SELESAI. File yang dihasilkan:")
for i, f in enumerate([
    "fig1_usia.png",
    "fig2_jk.png",
    "fig3_imt.png",
    "fig4_etiologi_alkohol.png",
    "fig5_daerah_asal.png",
    "fig6_ctp.png",
    "fig7_dashboard_demografi.png",
], start=1):
    print(f"  {i}. {f}")
print("=" * 60)