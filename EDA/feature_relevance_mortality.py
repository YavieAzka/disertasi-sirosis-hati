"""
====================================================================
Feature Relevance Analysis — Model Prediksi Outcome Klinis (Mortalitas)
Dataset Pasien Sirosis Hati (n=627, 137 fitur)
====================================================================
Target variabel  : Mortalitas (binary)
                   1 = Meninggal, 0 = Dipulangkan / APS
                   Sumber: kolom status_keluar
                   n_model = 396 (setelah eksklusi missing status_keluar)

Metode korelasi  :
  • Numerik → Point-Biserial Correlation (rpb) + p-value
  • Kategorik → Cramér's V + Chi-Square + p-value
  • Semua fitur → Mutual Information (MI, model-free, non-linear)

Output           :
  • Tabel ranking fitur per metode
  • Heatmap korelasi antar fitur numerik (multicollinearity check)
  • Bar chart importansi MI
  • Forest plot rpb + 95% CI
  • Scatter/violin per fitur terpilih
  • Dashboard ringkasan
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─── 0. KONFIGURASI ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#161b22",
    "axes.edgecolor":   "#2a3444", "axes.labelcolor": "#8b949e",
    "axes.titlecolor":  "#e6edf3", "xtick.color":     "#8b949e",
    "ytick.color":      "#8b949e", "text.color":      "#e6edf3",
    "grid.color":       "#2a3444", "grid.linewidth":  0.6,
    "font.family":      "DejaVu Sans", "font.size":   10,
    "axes.spines.top":  False,     "axes.spines.right": False,
})
BLUE  = "#58a6ff"; GREEN = "#3fb950"; GOLD  = "#d29922"
RED   = "#f78166"; PURPLE= "#bc8cff"; MUTED = "#8b949e"
DIM   = "#6e7681"; TEXT  = "#e6edf3"; BG    = "#0d1117"
SURFACE = "#161b22"

def style_ax(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_title(title, fontsize=10.5, fontweight="700", color=TEXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED, labelpad=5)
    if grid_axis:
        ax.grid(axis=grid_axis, lw=0.6, color="#2a3444", alpha=0.8)
        ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3444")
    ax.tick_params(colors=MUTED, labelsize=8.5)

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2: return 0.0
    chi2, _, _, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    return float(np.sqrt(phi2 / min(r-1, k-1)))

def pb_ci(r, n, alpha=0.05):
    """95% CI untuk point-biserial via Fisher z."""
    if abs(r) >= 1 or n < 4: return np.nan, np.nan
    z    = np.arctanh(r)
    se   = 1 / np.sqrt(n - 3)
    zc   = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


# ─── 1. LOAD & DEFINISI TARGET ───────────────────────────────────
print("=" * 65)
print("MEMUAT DATASET & MENDEFINISIKAN TARGET ...")
print("=" * 65)

df_raw = pd.read_excel("data_final_clean_manual.xlsx", sheet_name="Data")
N      = len(df_raw)

# Target: mortalitas binary dari status_keluar
valid_status = ['Dipulangkan', 'APS', 'Meninggal']
df = df_raw[df_raw['status_keluar'].isin(valid_status)].copy()
df['target'] = (df['status_keluar'] == 'Meninggal').astype(int)

N_MODEL  = len(df)
N_DEAD   = df['target'].sum()
N_ALIVE  = N_MODEL - N_DEAD
PREV     = N_DEAD / N_MODEL

print(f"\n  Dataset awal       : n={N}")
print(f"  Setelah eksklusi   : n={N_MODEL} (missing status_keluar dieksklusi)")
print(f"  Target — Meninggal : n={N_DEAD}  ({PREV*100:.1f}%)")
print(f"  Target — Hidup     : n={N_ALIVE} ({(1-PREV)*100:.1f}%)")
print(f"\n  [!] Class imbalance ratio: 1:{N_ALIVE/N_DEAD:.1f}")
print(f"      → Perlu dipertimbangkan SMOTE / class_weight saat training model")


# ─── 2. DEFINISI FITUR ───────────────────────────────────────────
# Fitur numerik (termasuk binary 0/1)
NUM_FEATURES = {
    # Demografi
    'usia'               : 'Usia',
    'imt'                : 'IMT',
    # Biokimia baseline
    'sgpt_baseline'      : 'SGPT Baseline',
    'sgot_baseline'      : 'SGOT Baseline',
    'albumin_baseline'   : 'Albumin Baseline',
    'bilirubin_baseline' : 'Bilirubin Baseline',
    'inr_baseline'       : 'INR Baseline',
    'kreatinin_baseline' : 'Kreatinin Baseline',
    'urea_baseline'      : 'Urea Baseline',
    'kalium_baseline'    : 'Kalium Baseline',
    'natrium_baseline'   : 'Natrium Baseline',
    'klorida_baseline'   : 'Klorida Baseline',
    # Fungsi ginjal
    'gfr'                : 'GFR (CKD-EPI)',
    # Komplikasi (binary)
    'komp_ascites'       : 'Asites',
    'komp_jaundice'      : 'Jaundice',
    'komp_eh'            : 'Ensefalopati Hepatikum',
    'komp_varises'       : 'Varises Esofagus',
    'komp_melena'        : 'Melena',
    'komp_sbp'           : 'SBP',
    # Komorbid (binary)
    'komor_dm'           : 'Diabetes Mellitus',
    'komor_ht'           : 'Hipertensi',
    'komor_pgk'          : 'Penyakit Ginjal Kronik',
    'komor_pneumonia'    : 'Pneumonia',
    'komor_sepsis'       : 'Sepsis',
    # Farmakologi
    # 'jumlah_obat'      : 'Jumlah Obat',  # dikeluarkan: all-NaN pada subset model
    'diuretik_ada'       : 'Diuretik (ada)',
    'betabloker_ada'     : 'Beta-bloker (ada)',
    'antibiotik_ada'     : 'Antibiotik (ada)',
    'analgetik_ada'      : 'Analgetik (ada)',
}

# Fitur kategorik
CAT_FEATURES = {
    'jk'                           : 'Jenis Kelamin',
    'etiologi'                     : 'Etiologi',
    'ctp'                          : 'Skor CTP',
    'gfr_kategori'                 : 'Kategori GFR',
    'imt_kategori'                 : 'Kategori IMT',
    'diuretik_kesesuaian'          : 'Kesesuaian Diuretik',
    'betabloker_kesesuaian'        : 'Kesesuaian Beta-bloker',
    'antibiotik_kesesuaian_gabungan': 'Kesesuaian Antibiotik',
    'analgetik_kesesuaian'         : 'Kesesuaian Analgetik',
}

# Filter hanya yang ada di dataset
NUM_FEATURES = {k: v for k, v in NUM_FEATURES.items() if k in df.columns}
CAT_FEATURES = {k: v for k, v in CAT_FEATURES.items() if k in df.columns}

print(f"\n  Fitur numerik/binary : {len(NUM_FEATURES)}")
print(f"  Fitur kategorik      : {len(CAT_FEATURES)}")


# ─── 3. ANALISIS A: POINT-BISERIAL (NUMERIK) ─────────────────────
print("\n" + "=" * 65)
print("ANALISIS A — Point-Biserial Correlation (Numerik vs Target)")
print("=" * 65)
print(f"\n  {'Fitur':<30} {'n':>5} {'rpb':>7} {'p':>9} {'95% CI':>18} {'Sig':>5} {'Missing':>8}")
print(f"  {'-'*82}")

pb_results = []
for feat, label in NUM_FEATURES.items():
    sub  = df[[feat, 'target']].dropna()
    n    = len(sub)
    miss = df[feat].isna().sum()
    if n < 20:
        continue
    r, p  = stats.pointbiserialr(sub['target'], sub[feat])
    lo, hi = pb_ci(r, n)
    ci_str = f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "—"
    pb_results.append({
        'feature': feat, 'label': label,
        'r': r, 'p': p, 'abs_r': abs(r),
        'ci_lo': lo, 'ci_hi': hi,
        'n': n, 'missing': miss,
        'sig': sig_label(p), 'type': 'numeric',
    })
    print(f"  {label:<30} {n:>5} {r:>+7.3f} {p:>9.4f} {ci_str:>18} "
          f"{sig_label(p):>5} {miss:>6}({miss/N_MODEL*100:.0f}%)")

pb_df = pd.DataFrame(pb_results).sort_values('abs_r', ascending=False)


# ─── 4. ANALISIS B: CRAMÉR'S V (KATEGORIK) ───────────────────────
print("\n" + "=" * 65)
print("ANALISIS B — Cramér's V (Kategorik vs Target)")
print("=" * 65)
print(f"\n  {'Fitur':<35} {'n':>5} {'V':>6} {'chi2':>8} {'p':>9} {'Sig':>5}")
print(f"  {'-'*70}")

cv_results = []
for feat, label in CAT_FEATURES.items():
    sub  = df[[feat, 'target']].dropna()
    n    = len(sub)
    miss = df[feat].isna().sum()
    if n < 20:
        continue
    v     = cramers_v(sub[feat], sub['target'])
    ct    = pd.crosstab(sub[feat], sub['target'])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    cv_results.append({
        'feature': feat, 'label': label,
        'v': v, 'chi2': chi2, 'p': p,
        'n': n, 'missing': miss,
        'sig': sig_label(p), 'type': 'categorical',
    })
    print(f"  {label:<35} {n:>5} {v:>6.3f} {chi2:>8.1f} {p:>9.4f} {sig_label(p):>5}")

cv_df = pd.DataFrame(cv_results).sort_values('v', ascending=False)


# ─── 5. ANALISIS C: MUTUAL INFORMATION ───────────────────────────
print("\n" + "=" * 65)
print("ANALISIS C — Mutual Information (MI, model-free, menangkap non-linearitas)")
print("=" * 65)

# Gabungkan semua fitur untuk MI
all_feat_cols = list(NUM_FEATURES.keys()) + list(CAT_FEATURES.keys())
df_mi = df[all_feat_cols + ['target']].copy()

# Encode kategorik
le = LabelEncoder()
for feat in CAT_FEATURES.keys():
    if feat in df_mi.columns:
        df_mi[feat] = df_mi[feat].astype(str)
        df_mi[feat] = le.fit_transform(df_mi[feat].fillna('_missing_'))

# Imputasi median untuk MI (MI butuh no-NaN)
imp = SimpleImputer(strategy='median')
X_mi = imp.fit_transform(df_mi[all_feat_cols])
y_mi = df_mi['target'].values

# Tentukan discrete_features sebagai list index (lebih aman dari boolean mask)
binary_cols   = [k for k in NUM_FEATURES if df[k].dropna().isin([0,1]).all()]
discrete_feat_names = binary_cols + list(CAT_FEATURES.keys())
discrete_idx  = [i for i, f in enumerate(all_feat_cols) if f in discrete_feat_names]

mi_scores = mutual_info_classif(
    X_mi, y_mi,
    discrete_features=discrete_idx,
    random_state=42
)

mi_df = pd.DataFrame({
    'feature': all_feat_cols,
    'label':   [NUM_FEATURES.get(f, CAT_FEATURES.get(f, f)) for f in all_feat_cols],
    'mi':      mi_scores,
    'type':    ['numeric' if f in NUM_FEATURES else 'categorical'
                for f in all_feat_cols],
}).sort_values('mi', ascending=False)

print(f"\n  {'Rank':<5} {'Fitur':<35} {'MI Score':>9} {'Tipe':>12}")
print(f"  {'-'*65}")
for i, (_, row) in enumerate(mi_df.iterrows(), 1):
    bar = '█' * int(row['mi'] * 60)
    print(f"  {i:<5} {row['label']:<35} {row['mi']:>9.4f} {row['type']:>12}")


# ─── 6. RANKING GABUNGAN ─────────────────────────────────────────
print("\n" + "=" * 65)
print("RANKING GABUNGAN — Semua Metode")
print("=" * 65)

# Merge semua metode berdasarkan feature
pb_rank = pb_df[['feature','label','abs_r','sig']].rename(
    columns={'abs_r':'rpb','sig':'sig_pb'})
cv_rank = cv_df[['feature','label','v','sig']].rename(
    columns={'v':'cramer_v','sig':'sig_cv'})
mi_rank = mi_df[['feature','label','mi']]

combined = mi_rank.merge(
    pb_rank[['feature','rpb','sig_pb']], on='feature', how='left'
).merge(
    cv_rank[['feature','cramer_v','sig_cv']], on='feature', how='left'
)

# Normalized rank score (rata-rata rank dari tiap metode)
combined['rank_mi']  = combined['mi'].rank(ascending=False)
combined['rank_rpb'] = combined['rpb'].rank(ascending=False, na_option='bottom')
combined['rank_cv']  = combined['cramer_v'].rank(ascending=False, na_option='bottom')
combined['avg_rank'] = combined[['rank_mi','rank_rpb','rank_cv']].mean(axis=1)
combined = combined.sort_values('avg_rank')

print(f"\n  {'Rank':<5} {'Fitur':<32} {'MI':>7} {'|rpb|':>7} {'V':>7} "
      f"{'AvgRank':>9} {'Sig (pb/cv)':>12}")
print(f"  {'-'*82}")
for i, (_, row) in enumerate(combined.head(20).iterrows(), 1):
    mi_v   = f"{row['mi']:.3f}"
    rpb_v  = f"{row['rpb']:.3f}" if not pd.isna(row.get('rpb')) else '—'
    cv_v   = f"{row['cramer_v']:.3f}" if not pd.isna(row.get('cramer_v')) else '—'
    sig_pb = str(row.get('sig_pb','—'))
    sig_cv = str(row.get('sig_cv','—'))
    sig_str = f"{sig_pb}/{sig_cv}"
    print(f"  {i:<5} {row['label']:<32} {mi_v:>7} {rpb_v:>7} {cv_v:>7} "
          f"{row['avg_rank']:>9.1f} {sig_str:>12}")

# Top features list
top_features = combined.head(15)['feature'].tolist()


# ─── 7. MULTICOLLINEARITY CHECK ──────────────────────────────────
print("\n" + "=" * 65)
print("CEK MULTIKOLINEARITAS antar Fitur Numerik Teratas")
print("(Jika |r|>0.7 antar dua fitur → pertimbangkan pilih salah satu)")
print("=" * 65)

top_num = [f for f in top_features if f in NUM_FEATURES][:12]
if len(top_num) > 1:
    corr_mat = df[top_num].corr(method='spearman')
    high_corr = []
    for i in range(len(top_num)):
        for j in range(i+1, len(top_num)):
            r_val = corr_mat.iloc[i,j]
            if abs(r_val) > 0.6:
                high_corr.append((top_num[i], top_num[j], r_val))

    if high_corr:
        print(f"\n  Pasangan dengan |r|>0.6:")
        for f1, f2, r_val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            l1 = NUM_FEATURES.get(f1, f1)
            l2 = NUM_FEATURES.get(f2, f2)
            flag = ' ← ⚠ |r|>0.7' if abs(r_val) > 0.7 else ''
            print(f"    {l1:<25} ↔ {l2:<25}: r={r_val:+.3f}{flag}")
    else:
        print("  Tidak ada multikolinearitas tinggi (|r|>0.6) antar fitur teratas.")


# ─── 8. RINGKASAN REKOMENDASI FITUR ─────────────────────────────
print("\n" + "=" * 65)
print("REKOMENDASI FITUR UNTUK MODEL MORTALITAS")
print("=" * 65)

# Fitur signifikan di minimal 2 metode
strong_features = []
for _, row in combined.iterrows():
    sig_count = 0
    if row['mi'] > 0.01: sig_count += 1
    if not pd.isna(row.get('rpb')) and row['rpb'] > 0.1: sig_count += 1
    if not pd.isna(row.get('cramer_v')) and row['cramer_v'] > 0.1: sig_count += 1
    if sig_count >= 2:
        strong_features.append(row)

strong_df = pd.DataFrame(strong_features).head(20)
print(f"\n  Fitur direkomendasikan (MI>0.01 + |rpb|>0.1 atau V>0.1):")
print(f"  {'Fitur':<32} {'MI':>7} {'|rpb|/V':>8} {'Kategori'}")
print(f"  {'-'*65}")
for _, row in strong_df.iterrows():
    score = row['rpb'] if pd.notna(row.get('rpb')) else row.get('cramer_v', 0)
    tipe  = row.get('type', '—')
    score = score if pd.notna(score) else 0.0
    print(f"  {row['label']:<32} {row['mi']:>7.4f} {score:>8.3f}   {tipe}")


# ─── 9. VISUALISASI ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("MEMBUAT VISUALISASI ...")
print("=" * 65)


# ════════════════════════════════════════════════════════════════
# FIG 1 — Forest Plot: Point-Biserial + CI (Top 20 fitur numerik)
# ════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(14, 10))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(SURFACE)

pb_sorted = pb_df.sort_values('r', ascending=True)
y_pos     = np.arange(len(pb_sorted))

for yi, (_, row) in enumerate(pb_sorted.iterrows()):
    is_sig = row['p'] < 0.05
    c = RED if row['r'] > 0 and is_sig else \
        GREEN if row['r'] < 0 and is_sig else DIM

    ax1.plot([row['ci_lo'], row['ci_hi']], [yi, yi],
             color=c, lw=2 if is_sig else 1, alpha=0.9 if is_sig else 0.4)
    ax1.scatter(row['r'], yi, color=c,
                s=70 if is_sig else 30,
                marker='D' if is_sig else 'o',
                zorder=5, edgecolors='white' if is_sig else 'none',
                linewidths=0.7)
    txt = (f"{row['r']:+.3f} {row['sig']}  "
           f"CI[{row['ci_lo']:.2f},{row['ci_hi']:.2f}]  n={row['n']}")
    ha = 'left' if row['r'] >= 0 else 'right'
    xpos = row['ci_hi'] + 0.01 if row['r'] >= 0 else row['ci_lo'] - 0.01
    ax1.text(xpos, yi, txt, va='center', fontsize=8,
             color=TEXT if is_sig else MUTED,
             fontweight='700' if is_sig else '400', ha=ha)

ax1.axvline(0, color=DIM, lw=1.3, ls='--', alpha=0.8)
ax1.axvspan(-0.1, 0.1, alpha=0.05, color=MUTED)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(pb_sorted['label'].values, fontsize=9, color=TEXT)
ax1.set_xlim(-0.65, 0.85)
ax1.set_xlabel("Point-Biserial r  (95% CI)\n← protective                          risk factor →",
               fontsize=9, color=MUTED)
ax1.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax1.set_axisbelow(True)
for sp in ax1.spines.values(): sp.set_edgecolor('#2a3444')
ax1.tick_params(colors=MUTED)
ax1.set_title(
    "Forest Plot: Point-Biserial Correlation — Fitur Numerik vs Mortalitas\n"
    "◆ signifikan (p<0.05)   ● tidak signifikan   merah=risk factor   hijau=protective",
    fontsize=11, fontweight='700', color=TEXT, pad=12)

# Legenda
patches = [
    mpatches.Patch(color=RED,   label='r>0, signifikan (risk factor)'),
    mpatches.Patch(color=GREEN, label='r<0, signifikan (protective)'),
    mpatches.Patch(color=DIM,   label='Tidak signifikan'),
]
ax1.legend(handles=patches, fontsize=9, facecolor='#1c2230',
           edgecolor='#2a3444', labelcolor=TEXT, loc='lower right')

fig1.tight_layout()
fig1.savefig('fig_feat1_forest_pb.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  ✓ fig_feat1_forest_pb.png")


# ════════════════════════════════════════════════════════════════
# FIG 2 — Bar Chart: Cramér's V (Fitur Kategorik)
# ════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(SURFACE)

cv_sorted = cv_df.sort_values('v', ascending=True)
y2        = np.arange(len(cv_sorted))
bar_colors = [GOLD if row['p'] < 0.05 else DIM
              for _, row in cv_sorted.iterrows()]

bars2 = ax2.barh(y2, cv_sorted['v'].values,
                 color=bar_colors, edgecolor='#0d1117', lw=0.4, alpha=0.88)
for bar, (_, row) in zip(bars2, cv_sorted.iterrows()):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f"V={row['v']:.3f} {row['sig']}  n={row['n']}",
             va='center', fontsize=9,
             color=TEXT if row['p'] < 0.05 else MUTED,
             fontweight='700' if row['p'] < 0.05 else '400')

ax2.axvline(0.1, color=MUTED, lw=1.2, ls=':', alpha=0.7, label='V=0.1 (efek kecil)')
ax2.axvline(0.3, color=GOLD,  lw=1.2, ls=':', alpha=0.7, label='V=0.3 (efek sedang)')
ax2.set_yticks(y2)
ax2.set_yticklabels(cv_sorted['label'].values, fontsize=9.5, color=TEXT)
ax2.set_xlim(0, cv_sorted['v'].max() * 1.5)
ax2.legend(fontsize=8.5, facecolor='#1c2230',
           edgecolor='#2a3444', labelcolor=TEXT)
ax2.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax2.set_axisbelow(True)
for sp in ax2.spines.values(): sp.set_edgecolor('#2a3444')
ax2.tick_params(colors=MUTED)
ax2.set_title("Cramér's V — Fitur Kategorik vs Mortalitas\n"
              "Emas = signifikan (p<0.05)",
              fontsize=11, fontweight='700', color=TEXT, pad=12)
ax2.set_xlabel("Cramér's V", fontsize=9, color=MUTED)

fig2.tight_layout()
fig2.savefig('fig_feat2_cramers_v.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  ✓ fig_feat2_cramers_v.png")


# ════════════════════════════════════════════════════════════════
# FIG 3 — Mutual Information Bar Chart (Top 20)
# ════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(13, 8))
fig3.patch.set_facecolor(BG)
ax3.set_facecolor(SURFACE)

mi_top = mi_df.head(20).sort_values('mi', ascending=True)
y3     = np.arange(len(mi_top))
colors3 = [BLUE if t == 'numeric' else PURPLE for t in mi_top['type']]

bars3 = ax3.barh(y3, mi_top['mi'].values,
                 color=colors3, edgecolor='#0d1117', lw=0.4, alpha=0.88)
for bar, (_, row) in zip(bars3, mi_top.iterrows()):
    ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f"{row['mi']:.4f}",
             va='center', fontsize=8.5, color=TEXT, fontweight='600')

ax3.set_yticks(y3)
ax3.set_yticklabels(mi_top['label'].values, fontsize=9.5, color=TEXT)
ax3.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax3.set_axisbelow(True)
for sp in ax3.spines.values(): sp.set_edgecolor('#2a3444')
ax3.tick_params(colors=MUTED)
ax3.set_xlabel("Mutual Information Score", fontsize=9, color=MUTED)
ax3.set_title("Mutual Information — Semua Fitur vs Mortalitas (Top 20)\n"
              "Biru = numerik/binary   Ungu = kategorik",
              fontsize=11, fontweight='700', color=TEXT, pad=12)

patches3 = [
    mpatches.Patch(color=BLUE,   label='Numerik / binary'),
    mpatches.Patch(color=PURPLE, label='Kategorik'),
]
ax3.legend(handles=patches3, fontsize=9, facecolor='#1c2230',
           edgecolor='#2a3444', labelcolor=TEXT)

fig3.tight_layout()
fig3.savefig('fig_feat3_mutual_info.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  ✓ fig_feat3_mutual_info.png")


# ════════════════════════════════════════════════════════════════
# FIG 4 — Heatmap Multikolinearitas (Top Fitur Numerik)
# ════════════════════════════════════════════════════════════════
top_num_all = pb_df.sort_values('abs_r', ascending=False).head(14)['feature'].tolist()
top_num_labels = [NUM_FEATURES[f] for f in top_num_all]
corr_mat = df[top_num_all].corr(method='spearman')

fig4, ax4 = plt.subplots(figsize=(13, 10))
fig4.patch.set_facecolor(BG)
ax4.set_facecolor(SURFACE)

cmap4 = LinearSegmentedColormap.from_list(
    "rb", [RED, "#1c2230", BLUE], N=256)
im4 = ax4.imshow(corr_mat.values, cmap=cmap4, vmin=-1, vmax=1, aspect='auto')

for i in range(len(top_num_all)):
    for j in range(len(top_num_all)):
        val = corr_mat.iloc[i, j]
        c_txt = '#0d1117' if abs(val) > 0.5 else TEXT
        fw = '700' if abs(val) > 0.6 and i != j else '400'
        ax4.text(j, i, f"{val:.2f}", ha='center', va='center',
                 fontsize=8, color=c_txt, fontweight=fw)
        if abs(val) > 0.6 and i != j:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                  fill=False, edgecolor=GOLD,
                                  lw=2, zorder=3)
            ax4.add_patch(rect)

ax4.set_xticks(range(len(top_num_all)))
ax4.set_yticks(range(len(top_num_all)))
ax4.set_xticklabels(top_num_labels, rotation=35, ha='right',
                    fontsize=9, color=TEXT)
ax4.set_yticklabels(top_num_labels, fontsize=9, color=TEXT)
ax4.tick_params(colors=MUTED)
ax4.set_title(
    "Heatmap Spearman — Multikolinearitas antar Fitur Numerik Terpilih\n"
    "Kotak emas = |r|>0.6 (risiko multikolinearitas)",
    fontsize=11, fontweight='700', color=TEXT, pad=12)

cbar4 = fig4.colorbar(im4, ax=ax4, fraction=0.03, pad=0.03)
cbar4.set_label("Spearman r", fontsize=9, color=MUTED)
cbar4.ax.tick_params(colors=MUTED, labelsize=8)
cbar4.outline.set_edgecolor('#2a3444')

fig4.tight_layout()
fig4.savefig('fig_feat4_multicollinearity.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  ✓ fig_feat4_multicollinearity.png")


# ════════════════════════════════════════════════════════════════
# FIG 5 — Violin Plot: Top 8 Fitur Numerik vs Target
# ════════════════════════════════════════════════════════════════
top8_num = pb_df.sort_values('abs_r', ascending=False).head(8)['feature'].tolist()
top8_labels = [NUM_FEATURES[f] for f in top8_num]

fig5, axes5 = plt.subplots(2, 4, figsize=(18, 9))
fig5.patch.set_facecolor(BG)
fig5.suptitle("Distribusi Top 8 Fitur Numerik per Outcome\n"
              "Hijau = Hidup (0)   Merah = Meninggal (1)",
              fontsize=13, fontweight='800', color=TEXT, y=1.01)

for idx, (feat, label) in enumerate(zip(top8_num, top8_labels)):
    ax = axes5.flatten()[idx]
    ax.set_facecolor(SURFACE)

    sub_feat = df[[feat, 'target']].dropna()
    v0 = sub_feat[sub_feat['target'] == 0][feat].values
    v1 = sub_feat[sub_feat['target'] == 1][feat].values

    if len(v0) > 2 and len(v1) > 2:
        vp = ax.violinplot([v0, v1], positions=[0, 1],
                           showmedians=True, showextrema=False)
        vp['bodies'][0].set_facecolor(GREEN); vp['bodies'][0].set_alpha(0.35)
        vp['bodies'][0].set_edgecolor(GREEN); vp['bodies'][0].set_linewidth(0.8)
        vp['bodies'][1].set_facecolor(RED);   vp['bodies'][1].set_alpha(0.35)
        vp['bodies'][1].set_edgecolor(RED);   vp['bodies'][1].set_linewidth(0.8)
        vp['cmedians'].set_color(GOLD); vp['cmedians'].set_linewidth(2)

        # Strip
        jit0 = np.random.uniform(-0.1, 0.1, len(v0))
        jit1 = np.random.uniform(-0.1, 0.1, len(v1))
        ax.scatter(0 + jit0, v0, color=GREEN, s=6, alpha=0.35, zorder=3)
        ax.scatter(1 + jit1, v1, color=RED,   s=6, alpha=0.35, zorder=3)

        # Median labels
        ax.text(0, np.median(v0) * 1.03, f"med={np.median(v0):.1f}",
                ha='center', fontsize=8, color=GREEN, fontweight='700')
        ax.text(1, np.median(v1) * 1.03, f"med={np.median(v1):.1f}",
                ha='center', fontsize=8, color=RED, fontweight='700')

    # Anotasi korelasi
    row_feat = pb_df[pb_df['feature'] == feat]
    if not row_feat.empty:
        row_feat = row_feat.iloc[0]
        ax.text(0.97, 0.97,
                f"r={row_feat['r']:+.3f} {row_feat['sig']}\nn={row_feat['n']}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                color=GOLD if row_feat['p'] < 0.05 else MUTED,
                fontweight='700' if row_feat['p'] < 0.05 else '400',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1c2230',
                          edgecolor='#2a3444', alpha=0.9))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Hidup (0)', 'Meninggal (1)'], fontsize=9, color=TEXT)
    style_ax(ax, label, '', feat.replace('_baseline','').upper(), 'y')

fig5.tight_layout()
fig5.savefig('fig_feat5_violin_top8.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  ✓ fig_feat5_violin_top8.png")


# ════════════════════════════════════════════════════════════════
# FIG 6 — Dashboard Ringkasan Feature Ranking
# ════════════════════════════════════════════════════════════════
fig6 = plt.figure(figsize=(18, 10))
fig6.patch.set_facecolor(BG)
gs6  = gridspec.GridSpec(2, 3, figure=fig6, hspace=0.45, wspace=0.40)
fig6.suptitle(
    "Dashboard Feature Relevance — Model Prediksi Mortalitas Pasien Sirosis Hati",
    fontsize=14, fontweight='800', color=TEXT, y=1.01)

# [0,0] Top 12 MI bar (horizontal)
ax60 = fig6.add_subplot(gs6[0, 0])
ax60.set_facecolor(SURFACE)
mi_top12 = mi_df.head(12).sort_values('mi', ascending=True)
colors60  = [BLUE if t == 'numeric' else PURPLE for t in mi_top12['type']]
bars60 = ax60.barh(range(len(mi_top12)), mi_top12['mi'].values,
                   color=colors60, edgecolor='#0d1117', lw=0.4, alpha=0.88)
for bar, (_, row) in zip(bars60, mi_top12.iterrows()):
    ax60.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
              f"{row['mi']:.3f}", va='center', fontsize=8, color=TEXT)
ax60.set_yticks(range(len(mi_top12)))
ax60.set_yticklabels(mi_top12['label'].values, fontsize=8.5, color=TEXT)
ax60.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax60.set_axisbelow(True)
for sp in ax60.spines.values(): sp.set_edgecolor('#2a3444')
ax60.tick_params(colors=MUTED)
style_ax(ax60, 'Mutual Information (Top 12)', 'MI Score', '', grid_axis=None)

# [0,1] Top 12 |rpb| bar
ax61 = fig6.add_subplot(gs6[0, 1])
ax61.set_facecolor(SURFACE)
pb_top12 = pb_df.head(12).sort_values('abs_r', ascending=True)
colors61  = [RED if r > 0 else GREEN for r in pb_top12['r']]
bars61 = ax61.barh(range(len(pb_top12)), pb_top12['abs_r'].values,
                   color=colors61, edgecolor='#0d1117', lw=0.4, alpha=0.88)
for bar, (_, row) in zip(bars61, pb_top12.iterrows()):
    ax61.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
              f"{row['r']:+.3f} {row['sig']}", va='center', fontsize=8,
              color=TEXT if row['p']<0.05 else MUTED)
ax61.set_yticks(range(len(pb_top12)))
ax61.set_yticklabels(pb_top12['label'].values, fontsize=8.5, color=TEXT)
ax61.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax61.set_axisbelow(True)
for sp in ax61.spines.values(): sp.set_edgecolor('#2a3444')
ax61.tick_params(colors=MUTED)
style_ax(ax61, '|Point-Biserial r| (Top 12)\nMerah=risk, Hijau=protective',
         '|rpb|', '', grid_axis=None)

# [0,2] Cramér's V bar
ax62 = fig6.add_subplot(gs6[0, 2])
ax62.set_facecolor(SURFACE)
cv_s = cv_df.sort_values('v', ascending=True)
colors62 = [GOLD if p < 0.05 else DIM for p in cv_s['p']]
bars62 = ax62.barh(range(len(cv_s)), cv_s['v'].values,
                   color=colors62, edgecolor='#0d1117', lw=0.4, alpha=0.88)
for bar, (_, row) in zip(bars62, cv_s.iterrows()):
    ax62.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
              f"V={row['v']:.3f} {row['sig']}", va='center', fontsize=8,
              color=TEXT if row['p']<0.05 else MUTED)
ax62.set_yticks(range(len(cv_s)))
ax62.set_yticklabels(cv_s['label'].values, fontsize=8.5, color=TEXT)
ax62.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax62.set_axisbelow(True)
for sp in ax62.spines.values(): sp.set_edgecolor('#2a3444')
ax62.tick_params(colors=MUTED)
style_ax(ax62, "Cramér's V — Kategorik\nEmas = signifikan",
         "V", '', grid_axis=None)

# [1,0:2] Combined ranking bar (avg rank score)
ax63 = fig6.add_subplot(gs6[1, :2])
ax63.set_facecolor(SURFACE)
top20_combined = combined.head(20).sort_values('avg_rank', ascending=False)
colors63 = [BLUE if mi_df[mi_df['feature']==f]['type'].values[0] == 'numeric'
            else PURPLE
            for f in top20_combined['feature']]
bars63 = ax63.barh(range(len(top20_combined)),
                   1/top20_combined['avg_rank'].values,  # inverse rank = higher = better
                   color=colors63, edgecolor='#0d1117', lw=0.4, alpha=0.88)
ax63.set_yticks(range(len(top20_combined)))
ax63.set_yticklabels(top20_combined['label'].values, fontsize=8.5, color=TEXT)
ax63.grid(axis='x', lw=0.6, color='#2a3444', alpha=0.8)
ax63.set_axisbelow(True)
for sp in ax63.spines.values(): sp.set_edgecolor('#2a3444')
ax63.tick_params(colors=MUTED)
style_ax(ax63, 'Ranking Gabungan (MI + |rpb| + V) — Top 20 Fitur\n'
               'Biru=numerik/binary  Ungu=kategorik  (bar = inverse avg rank)',
         'Skor (inverse avg rank)', '', grid_axis=None)

# [1,2] Ringkasan teks
ax64 = fig6.add_subplot(gs6[1, 2])
ax64.set_facecolor(BG)
for sp in ax64.spines.values(): sp.set_visible(False)
ax64.set_xticks([]); ax64.set_yticks([])

top5 = combined.head(5)
lines64 = [
    ("TARGET: MORTALITAS", TEXT, 10, True),
    (f"n={N_MODEL}  pos={N_DEAD} ({PREV*100:.0f}%)", MUTED, 8.5, False),
    ("", MUTED, 8, False),
    ("TOP 5 FITUR (Gabungan):", TEXT, 9.5, True),
] + [
    (f"  {i+1}. {row['label']}", GOLD if i < 3 else TEXT, 9, i < 3)
    for i, (_, row) in enumerate(top5.iterrows())
] + [
    ("", MUTED, 8, False),
    ("⚠ Class imbalance:", TEXT, 9, True),
    (f"  Hidup:{N_ALIVE} vs Meninggal:{N_DEAD}", MUTED, 8.5, False),
    (f"  Ratio 1:{N_ALIVE/N_DEAD:.1f}", RED, 8.5, True),
    ("  → Gunakan SMOTE /", MUTED, 8.5, False),
    ("    class_weight='balanced'", MUTED, 8.5, False),
    ("", MUTED, 8, False),
    ("⚠ Missing data tinggi:", TEXT, 9, True),
    ("  INR (50%), Bilirubin (47%)", MUTED, 8.5, False),
    ("  → Perlu imputasi sebelum", MUTED, 8.5, False),
    ("    training model", MUTED, 8.5, False),
]
y64 = 0.97
for txt, col, sz, bold in lines64:
    ax64.text(0.03, y64, txt, transform=ax64.transAxes,
              fontsize=sz, color=col,
              fontweight='700' if bold else '400',
              va='top', fontfamily='monospace')
    y64 -= 0.062

fig6.savefig('fig_feat6_dashboard.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
print("  ✓ fig_feat6_dashboard.png")


# ─── 10. RINGKASAN AKHIR ─────────────────────────────────────────
print("\n" + "=" * 65)
print("RINGKASAN UNTUK MODEL MORTALITAS")
print("=" * 65)

print(f"""
  TARGET   : Mortalitas (binary) — {N_DEAD}/{N_MODEL} positif ({PREV*100:.1f}%)
  METODE   : Point-Biserial + Cramér's V + Mutual Information

  TOP FITUR BERDASARKAN RANKING GABUNGAN:
""")
for i, (_, row) in enumerate(combined.head(15).iterrows(), 1):
    mi_s  = f"MI={row['mi']:.3f}"
    rpb_s = f"|r|={row['rpb']:.3f}" if not pd.isna(row.get('rpb')) else ""
    cv_s  = f"V={row['cramer_v']:.3f}" if not pd.isna(row.get('cramer_v')) else ""
    scores = "  ".join([s for s in [mi_s, rpb_s, cv_s] if s])
    print(f"  {i:>2}. {row['label']:<32} {scores}")

print(f"""
  CATATAN UNTUK PREPROCESSING MODEL:
  1. Class imbalance: ratio 1:{N_ALIVE/N_DEAD:.1f} → gunakan SMOTE atau
     class_weight='balanced' saat training
  2. Missing data tinggi pada beberapa fitur utama:
     INR (50%), Bilirubin (47%), Albumin (39%), SGPT/SGOT (34%)
     → Imputasi (median/KNN imputer) diperlukan
  3. Multikolinearitas: cek pasangan fitur dengan |r|>0.7
     sebelum memasukkan ke model linear/logistik
  4. GFR adalah variabel turunan (CKD-EPI dari kreatinin)
     → posisikan sebagai derived feature dalam dokumentasi
""")

print("=" * 65)
print("SELESAI. File yang dihasilkan:")
for i, f in enumerate([
    "fig_feat1_forest_pb.png       — Forest plot Point-Biserial + CI",
    "fig_feat2_cramers_v.png       — Bar chart Cramér's V kategorik",
    "fig_feat3_mutual_info.png     — Bar chart Mutual Information Top 20",
    "fig_feat4_multicollinearity.png — Heatmap multikolinearitas",
    "fig_feat5_violin_top8.png     — Violin plot Top 8 fitur vs target",
    "fig_feat6_dashboard.png       — Dashboard ranking gabungan",
], start=1):
    print(f"  {i}. {f}")
print("=" * 65)