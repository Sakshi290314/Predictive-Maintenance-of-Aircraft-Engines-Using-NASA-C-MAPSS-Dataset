# Create comparison visualizations
print("\n" + "=" * 50)
print("Creating Visualizations")
print("=" * 50)

# Create figure with 3 subplots for comparing predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Random Forest
axes[0].scatter(y_val, rf_val_pred, alpha=0.5, s=10)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('True RUL (cycles)', fontsize=11)
axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[0].set_title(f'Random Forest\nRMSE: {rf_val_rmse:.2f} | R²: {rf_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# XGBoost
axes[1].scatter(y_val, xgb_val_pred, alpha=0.5, s=10, color='orange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('True RUL (cycles)', fontsize=11)
axes[1].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[1].set_title(f'XGBoost\nRMSE: {xgb_val_rmse:.2f} | R²: {xgb_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# LightGBM
axes[2].scatter(y_val, lgb_val_pred, alpha=0.5, s=10, color='green')
axes[2].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[2].set_xlabel('True RUL (cycles)', fontsize=11)
axes[2].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[2].set_title(f'LightGBM\nRMSE: {lgb_val_rmse:.2f} | R²: {lgb_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\Saksh\Downloads\ml_predictions.png', dpi=300, bbox_inches='tight')
print(r"\nSaved: 'C:\Users\Saksh\Downloads\ml_predictions.png'")
plt.close()

# Create feature importance visualization using XGBoost
print("\nCreating feature importance chart...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\Saksh\Downloads\feature_importance.png', dpi=300, bbox_inches='tight')
print(r"Saved: 'C:\Users\Saksh\Downloads\feature_importance.png'")
plt.close()

# Create model comparison table
print("\n" + "=" * 50)
print("Model Comparison Summary")
print("=" * 50)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'Train RMSE': [rf_train_rmse, xgb_train_rmse, lgb_train_rmse],
    'Val RMSE': [rf_val_rmse, xgb_val_rmse, lgb_val_rmse],
    'Train MAE': [rf_train_mae, xgb_train_mae, lgb_train_mae],
    'Val MAE': [rf_val_mae, xgb_val_mae, lgb_val_mae],
    'Val R²': [rf_val_r2, xgb_val_r2, lgb_val_r2]
})

print("\n" + comparison.to_string(index=False))

# Save comparison results to CSV
comparison.to_csv(r'C:\Users\Saksh\Downloads\model_comparison.csv', index=False)
print(r"\nSaved: 'C:\Users\Saksh\Downloads\model_comparison.csv'")

# Display summary
print("\n" + "=" * 50)
print("Phase 3 Complete!")
print("=" * 50)

best_model = comparison.loc[comparison['Val RMSE'].idxmin(), 'Model']
best_rmse = comparison['Val RMSE'].min()

print(f"\nBest Model: {best_model}")
print(f"Best Validation RMSE: {best_rmse:.2f} cycles")
print(f"\nFiles created:")
print(f"  1. random_forest.pkl")
print(f"  2. xgboost.pkl")
print(f"  3. lightgbm.pkl")
print(f"  4. ml_predictions.png")
print(f"  5. feature_importance.png")
print(f"  6. model_comparison.csv")
print("=" * 50)


"""
ML MODEL COMPARISON DASHBOARD
Clean rewrite — no overlap, no wasted space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Palette & style ──────────────────────────────────────────────────────────
COLORS = {
    "Random Forest": "#2196F3",   # blue
    "XGBoost":       "#F44336",   # red
    "LightGBM":      "#FF9800",   # orange
}
GOLD   = "#FFD700"
BG     = "#F4F6FA"
DARK   = "#1C1C2E"
GRID_C = "#CCCCCC"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "grid.color":        GRID_C,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

# ═══════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════
train_df     = pd.read_csv(r'C:\Users\Saksh\Downloads\train_processed.csv')
feature_cols = joblib.load(r'C:\Users\Saksh\Downloads\feature_columns.pkl')
X = train_df[feature_cols]
y = train_df['RUL']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model  = joblib.load(r'C:\Users\Saksh\Downloads\random_forest.pkl')
xgb_model = joblib.load(r'C:\Users\Saksh\Downloads\xgboost.pkl')
lgb_model = joblib.load(r'C:\Users\Saksh\Downloads\lightgbm.pkl')
models = {"Random Forest": rf_model, "XGBoost": xgb_model, "LightGBM": lgb_model}

# ═══════════════════════════════════════════════════════════════
# 2. EVALUATE
# ═══════════════════════════════════════════════════════════════
baseline_rmse = np.sqrt(mean_squared_error(y_val, np.full(len(y_val), y_train.mean())))

rows = []
for name, mdl in models.items():
    tr_p = mdl.predict(X_train);  vl_p = mdl.predict(X_val)
    tr_r = np.sqrt(mean_squared_error(y_train, tr_p))
    vl_r = np.sqrt(mean_squared_error(y_val,   vl_p))
    rows.append({
        "Model":      name,
        "tr_rmse":    tr_r,
        "vl_rmse":    vl_r,
        "vl_mae":     mean_absolute_error(y_val, vl_p),
        "vl_r2":      r2_score(y_val, vl_p),
        "gap":        ((vl_r - tr_r) / tr_r) * 100,
        "improvement":((baseline_rmse - vl_r) / baseline_rmse) * 100,
    })

df = pd.DataFrame(rows).sort_values("vl_rmse").reset_index(drop=True)
names  = df["Model"].tolist()
clrs   = [COLORS[n] for n in names]

# Save CSV
df.rename(columns={
    "tr_rmse":"Train RMSE","vl_rmse":"Val RMSE","vl_mae":"Val MAE",
    "vl_r2":"Val R2","gap":"Overfitting Gap (%)","improvement":"Improvement (%)"
}).to_csv(r'C:\Users\Saksh\Downloads\FINAL_ML_COMPARISON_TABLE.csv', index=False)
print(df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 3. DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════
def ylim_zoom(vals, top=0.30, bot=0.08):
    lo, hi = min(vals), max(vals)
    s = (hi - lo) or abs(hi)*0.1 or 1
    return lo - bot*s, hi + top*s

def add_value_labels(ax, bars, values, fmt, all_vals):
    """White label inside bar if tall enough, dark label above if short."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    for bar, v in zip(bars, values):
        h  = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        if h > span * 0.22:                        # tall bar → inside white
            ax.text(cx, bar.get_y() + h * 0.85,
                    fmt.format(v), ha='center', va='top',
                    fontsize=8.5, fontweight='bold', color='white', zorder=7)
        else:                                       # short bar → above dark
            ax.text(cx, bar.get_y() + h + span * 0.02,
                    fmt.format(v), ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=DARK, zorder=7)

def gold_outline(ax, bar):
    lo = ax.get_ylim()[0]
    h  = bar.get_height()
    ax.add_patch(FancyBboxPatch(
        (bar.get_x()-0.025, lo+0.001),
        bar.get_width()+0.05, h - lo,
        boxstyle="round,pad=0.005",
        linewidth=2.5, edgecolor=GOLD, facecolor="none", zorder=8))

def vbar(ax, vals, title, ylabel, fmt,
         higher_better=False, extra_fn=None):
    """Generic vertical bar panel."""
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=clrs, width=0.5,
                  edgecolor='white', linewidth=1.3, zorder=3)
    ax.set_ylim(ylim_zoom(vals))
    add_value_labels(ax, bars, vals, fmt, vals)
    best = int(np.argmax(vals) if higher_better else np.argmin(vals))
    gold_outline(ax, bars[best])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha='right',
                       rotation_mode='anchor', fontsize=8, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8, color='#555')
    ax.set_title(title, fontsize=10.5, fontweight='bold',
                 color=DARK, pad=10, linespacing=1.5)
    ax.tick_params(axis='x', length=0, pad=3)
    if extra_fn:
        extra_fn(ax)

# ═══════════════════════════════════════════════════════════════
# 4.  DASHBOARD FIGURE
#     Strategy: let matplotlib place everything, then call
#     tight_layout() so there is NEVER wasted space.
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(
    2, 3,
    figsize=(19, 10),          # wide enough, not too tall
    facecolor=BG,
)
fig.patch.set_facecolor(BG)

# Reserve top 8% for two title lines, bottom 6% for legend
fig.subplots_adjust(top=0.88, bottom=0.10,
                    left=0.06, right=0.97,
                    hspace=0.60, wspace=0.35)

fig.text(0.5, 0.96,
         "ML Model Performance Dashboard",
         ha='center', fontsize=19, fontweight='bold', color=DARK)
fig.text(0.5, 0.915,
         "Random Forest  ·  XGBoost  ·  LightGBM  —  RUL Prediction",
         ha='center', fontsize=10, color='#666')

ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

# ── Panel 1 : Validation RMSE ───────────────────────────────
vbar(ax1, df["vl_rmse"].tolist(),
     "Validation RMSE\n(Lower is Better)", "RMSE (cycles)", "{:.2f}")

# ── Panel 2 : Validation R² ─────────────────────────────────
vbar(ax2, df["vl_r2"].tolist(),
     "Model Accuracy  R²\n(Higher is Better)", "R² Score", "{:.4f}",
     higher_better=True)

# ── Panel 3 : Train vs Val RMSE ─────────────────────────────
xpos = np.arange(len(names));  w = 0.30
tr_list = df["tr_rmse"].tolist();  vl_list = df["vl_rmse"].tolist()
light   = [c+"88" for c in clrs]   # faded for train bars
b_tr = ax3.bar(xpos - w/2, tr_list, width=w, color=light,
               edgecolor='white', linewidth=1.1, label="Train", zorder=3)
b_vl = ax3.bar(xpos + w/2, vl_list, width=w, color=clrs,
               edgecolor='white', linewidth=1.1, label="Val",   zorder=3)
ax3.set_ylim(ylim_zoom(tr_list + vl_list, top=0.32))
add_value_labels(ax3, list(b_tr)+list(b_vl), tr_list+vl_list, "{:.1f}", tr_list+vl_list)
ax3.set_xticks(xpos)
ax3.set_xticklabels(names, rotation=12, ha='right',
                    rotation_mode='anchor', fontsize=8, fontweight='bold')
ax3.set_ylabel("RMSE (cycles)", fontsize=8, color='#555')
ax3.set_title("Train vs Validation RMSE\n(Overfitting Check)",
              fontsize=10.5, fontweight='bold', color=DARK, pad=10, linespacing=1.5)
ax3.legend(fontsize=8, framealpha=0.75, loc='upper right')
ax3.tick_params(axis='x', length=0, pad=3)

# ── Panel 4 : Overfitting Gap ───────────────────────────────
def gap_extras(ax):
    ax.axhline(10, color='#E53935', lw=1.4, ls='--', label='10% threshold', zorder=5)
    ax.legend(fontsize=7.5, framealpha=0.75, loc='upper right')
vbar(ax4, df["gap"].tolist(),
     "Overfitting Gap (%)\n(Lower is Better)", "Gap (%)", "{:.1f}%",
     extra_fn=gap_extras)

# ── Panel 5 : Validation MAE ────────────────────────────────
vbar(ax5, df["vl_mae"].tolist(),
     "Mean Absolute Error\n(Lower is Better)", "MAE (cycles)", "{:.2f}")

# ── Panel 6 : Improvement Over Baseline ─────────────────────
def imp_extras(ax):
    ax.axhline(25, color='#FFA000', lw=1.4, ls='--', label='25% target', zorder=5)
    ax.axhline(50, color='#388E3C', lw=1.4, ls='--', label='50% target', zorder=5)
    ax.legend(fontsize=7.5, framealpha=0.75, loc='lower right')
vbar(ax6, df["improvement"].tolist(),
     "Improvement Over Baseline\n(Higher is Better)", "Improvement (%)", "{:.1f}%",
     higher_better=True, extra_fn=imp_extras)

# ── Legend strip ─────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[n], label=n) for n in names]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01),
           edgecolor='#ccc')

fig.savefig(r'C:\Users\Saksh\Downloads\ML_DASHBOARD.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print("\n  ML_DASHBOARD.png saved!")


# ═══════════════════════════════════════════════════════════════
# 5. RANKING TABLE
# ═══════════════════════════════════════════════════════════════
fig2, ax_t = plt.subplots(figsize=(11, 3.5), facecolor=BG)
ax_t.set_facecolor(BG);  ax_t.axis('off')

col_labels = ["Rank", "Model", "Val RMSE", "Val MAE", "R² (%)", "Overfit Gap"]
rank_bg    = [GOLD, "#C0C0C0", "#CD7F32"]
col_vals   = []
for i, row in df.iterrows():
    col_vals.append([
        f"  #{i+1}",
        row["Model"],
        f"{row['vl_rmse']:.2f}",
        f"{row['vl_mae']:.2f}",
        f"{row['vl_r2']*100:.2f}%",
        f"{row['gap']:.1f}%",
    ])

tbl = ax_t.table(cellText=col_vals, colLabels=col_labels,
                 cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False);  tbl.set_fontsize(11)
tbl.auto_set_column_width(list(range(len(col_labels))))

for j in range(len(col_labels)):     # header row
    c = tbl[0, j]
    c.set_facecolor(DARK);  c.set_text_props(color='white', fontweight='bold')
    c.set_edgecolor('white')

for i in range(1, len(col_vals)+1): # data rows
    mn = col_vals[i-1][1]
    for j in range(len(col_labels)):
        c = tbl[i, j]
        c.set_edgecolor("#DDDDDD")
        c.set_facecolor(COLORS.get(mn,"#fff") + "20")
        if j == 0:
            c.set_facecolor(rank_bg[i-1] if i<=3 else "#eee")
            c.set_text_props(fontweight='bold', fontsize=13)
        if j == 1:
            c.set_text_props(fontweight='bold')

fig2.suptitle("Performance Comparison — ML Models",
              fontsize=14, fontweight='bold', color=DARK, y=1.02)
fig2.savefig(r'C:\Users\Saksh\Downloads\ML_RANKING_TABLE.png',
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print("  ML_RANKING_TABLE.png saved!")


# ═══════════════════════════════════════════════════════════════
# 6. HORIZONTAL IMPROVEMENT CHART
# ═══════════════════════════════════════════════════════════════
df_i  = df.sort_values("improvement", ascending=True).reset_index(drop=True)
i_v   = df_i["improvement"].tolist()
i_n   = df_i["Model"].tolist()
i_c   = [COLORS[n] for n in i_n]

fig3, ax_h = plt.subplots(figsize=(9, 3.5), facecolor=BG)
ax_h.set_facecolor(BG)

yp = np.arange(len(i_n))
bh = ax_h.barh(yp, i_v, color=i_c, height=0.45,
               edgecolor='white', linewidth=1.4, zorder=3)

sp = max(i_v) - min(i_v) or 1
for bar, val in zip(bh, i_v):
    cy = bar.get_y() + bar.get_height()/2
    if val > sp*0.3:
        ax_h.text(val*0.96, cy, f"{val:.1f}%", va='center', ha='right',
                  fontsize=10, fontweight='bold', color='white', zorder=6)
    else:
        ax_h.text(val + sp*0.02, cy, f"{val:.1f}%", va='center', ha='left',
                  fontsize=10, fontweight='bold', color=DARK, zorder=6)

ax_h.axvline(25, color='#FFA000', lw=1.8, ls='--', label='25%', zorder=5)
ax_h.axvline(50, color='#388E3C', lw=1.8, ls='--', label='50%', zorder=5)
ax_h.set_yticks(yp);  ax_h.set_yticklabels(i_n, fontsize=11, fontweight='bold')
ax_h.set_xlabel("Improvement over Baseline (%)", fontsize=9, color='#555')
ax_h.set_title("Model Performance Improvement (vs Mean Baseline)",
               fontsize=12, fontweight='bold', color=DARK, pad=10)
ax_h.set_xlim(0, max(i_v)*1.20)
ax_h.legend(fontsize=9, framealpha=0.8, loc='lower right')
ax_h.tick_params(axis='y', length=0)
ax_h.spines['left'].set_visible(False)

fig3.tight_layout()
fig3.savefig(r'C:\Users\Saksh\Downloads\ML_IMPROVEMENT_CHART.png',
             dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print("  ML_IMPROVEMENT_CHART.png saved!")

print("\n" + "="*55)
print("All files saved to C:\\Users\\Saksh\\Downloads\\")
print("="*55)
