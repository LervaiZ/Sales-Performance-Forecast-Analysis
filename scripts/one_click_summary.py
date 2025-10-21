#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click summary for Sales Performance Forecast & Analysis
- Fits OLS on log(Total) with HC3 robust SE
- Outputs executive_summary.txt, significant_terms.csv, diagnostics plot, example_prediction.csv
Usage:
  python scripts/one_click_summary.py --input path/to/sales.csv --outdir .
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

RENAME_MAP = {
    'Unit price':'Unit_price',
    'Product line':'Product_line',
    'Customer type':'Customer_type',
    'Tax 5%':'Tax_5',
    'gross income':'gross_income',
    'gross margin percentage':'gross_margin_percentage'
}

FORMULA = """
np.log(Total) ~ Quantity + Unit_price + Rating
              + C(City) + C(Product_line) + C(Customer_type) + C(Payment)
              + C(Month)
""".strip()

NUM_COLS  = ['Quantity', 'Unit_price', 'Rating']
CAT_COLS  = ['City', 'Product_line', 'Customer_type', 'Payment', 'Month']

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # rename friendly columns if present
    df = df.rename(columns={k:v for k,v in RENAME_MAP.items() if k in df.columns})
    if 'Month' not in df.columns and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
    return df

def fit_model(df: pd.DataFrame):
    model = smf.ols(formula=FORMULA, data=df).fit()
    robust = model.get_robustcov_results(cov_type='HC3')
    return model, robust

def compute_metrics(df: pd.DataFrame, robust) -> dict:
    resid  = np.asarray(robust.resid, dtype=float)
    fitted = np.asarray(robust.fittedvalues, dtype=float)
    rmse_log = float(np.sqrt(np.mean(resid**2)))
    smear = float(np.mean(np.exp(resid)))
    y_true = df['Total'].astype(float).to_numpy()
    y_pred = np.exp(fitted) * smear
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    rmse_original = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))
    return {
        "adj_r2": float(robust.rsquared_adj),
        "rmse_log": rmse_log,
        "rmse_original": rmse_original,
        "smear": smear,
        "fitted": fitted,
        "resid": resid
    }

def significant_terms(robust) -> pd.DataFrame:
    term_names = getattr(robust.model, "exog_names", [f"param_{i}" for i in range(len(robust.params))])
    params = np.asarray(robust.params, dtype=float)
    pvals  = np.asarray(robust.pvalues, dtype=float)
    df_sig = pd.DataFrame({"term": term_names, "coef": params, "pval": pvals})
    df_sig = df_sig.query("term not in ['Intercept','const']")
    df_sig["effect_pct"] = (np.exp(df_sig["coef"]) - 1.0) * 100.0
    return df_sig[df_sig["pval"] < 0.05].sort_values("effect_pct", ascending=False)

def plot_diagnostics(resid: np.ndarray, fitted: np.ndarray, outpath: str):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    sns.scatterplot(x=fitted, y=resid, ax=axes[0], alpha=0.6, s=25)
    axes[0].axhline(0, color='red', ls='--', lw=1)
    axes[0].set_title('Residuals vs Fitted (log-scale)')
    axes[0].set_xlabel('Fitted (log Total)')
    axes[0].set_ylabel('Residuals')
    sm.ProbPlot(resid).qqplot(line='45', ax=axes[1])
    axes[1].set_title('QQ Plot')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    # Choose medians for numeric, mode for categoricals (valid categories from data)
    sample = {}
    for c in NUM_COLS:
        if c in df.columns:
            sample[c] = float(df[c].median())
    for c in CAT_COLS:
        if c in df.columns:
            try:
                sample[c] = df[c].mode(dropna=True).iloc[0]
            except Exception:
                sample[c] = df[c].dropna().iloc[0]
    # defaults as fallback
    defaults = {
        'Quantity': 6, 'Unit_price': 60, 'Rating': 8.5,
        'City': 'Naypyitaw', 'Product_line': 'Food and beverages',
        'Customer_type': 'Member', 'Payment': 'Credit card', 'Month': 1
    }
    for k, v in defaults.items():
        sample.setdefault(k, v)
    return pd.DataFrame([sample])

def predict_with_ci(robust, sample: pd.DataFrame, smear: float) -> pd.DataFrame:
    sf = robust.get_prediction(sample).summary_frame(alpha=0.05)
    out = pd.DataFrame({
        'pred_total':  np.exp(sf['mean']) * smear,
        'CI_low_95':   np.exp(sf['mean_ci_lower']) * smear,
        'CI_high_95':  np.exp(sf['mean_ci_upper']) * smear
    })
    return out

def write_summary_txt(path: str, metrics: dict, sig_df: pd.DataFrame, example_pred: pd.DataFrame):
    lines = []
    lines.append("Model: OLS on log(Total) with HC3 robust SE")
    lines.append(f"Adj R² = {metrics['adj_r2']:.3f} | RMSE(log) = {metrics['rmse_log']:.4f} | RMSE(original) = {metrics['rmse_original']:.2f}")
    lines.append("")
    lines.append("Top significant effects (p < 0.05):")
    if len(sig_df) == 0:
        lines.append("- (none)")
    else:
        for _, r in sig_df.iterrows():
            term = r['term']
            eff  = r['effect_pct']
            if term.startswith('C('):
                lines.append(f"- {term}: {eff:+.2f}%")
            else:
                lines.append(f"- {term} (per +1): {eff:+.2f}%")
    lines.append("")
    if not example_pred.empty:
        pt = float(example_pred['pred_total'].iloc[0])
        lo = float(example_pred['CI_low_95'].iloc[0])
        hi = float(example_pred['CI_high_95'].iloc[0])
        lines.append(f"Example prediction (original scale, 95% CI): {pt:.2f} ({lo:.2f} – {hi:.2f})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to sales CSV")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)
    model, robust = fit_model(df)
    metrics = compute_metrics(df, robust)

    sig = significant_terms(robust)
    sig_out = os.path.join(args.outdir, "significant_terms.csv")
    sig.to_csv(sig_out, index=False)

    assets_dir = os.path.join(args.outdir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    diag_path = os.path.join(assets_dir, "diagnostics_resid_qq.png")
    plot_diagnostics(metrics["resid"], metrics["fitted"], diag_path)

    sample = build_sample(df)
    example_pred = predict_with_ci(robust, sample, metrics["smear"])
    example_pred.to_csv(os.path.join(args.outdir, "example_prediction.csv"), index=False)

    summary_path = os.path.join(args.outdir, "executive_summary.txt")
    write_summary_txt(summary_path, metrics, sig, example_pred)

    print("Done.")
    print(f"- executive_summary.txt -> {summary_path}")
    print(f"- significant_terms.csv  -> {sig_out}")
    print(f"- diagnostics plot       -> {diag_path}")
    print(f"- example_prediction.csv -> {os.path.join(args.outdir, 'example_prediction.csv')}")

if __name__ == "__main__":
    main()
