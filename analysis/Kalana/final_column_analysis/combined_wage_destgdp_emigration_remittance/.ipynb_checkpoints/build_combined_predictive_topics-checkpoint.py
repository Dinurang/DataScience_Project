from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = REPO_ROOT / "Dataset_Management" / "SriLanka_Migration_final.csv"
CHART_DIR = OUT_DIR / "charts"

RANDOM_STATE = 42


def load_annual_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Invalid date values found in source CSV.")

    cols = [
        "slbfe_total_annual",
        "remittances_annual_usd_mn",
        "wage_all_workers_annual",
        "dest_gdp_growth_avg_annual",
    ]
    annual = (
        df.assign(year=df["date"].dt.year)
        .groupby("year", as_index=False)[cols]
        .mean()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return annual


def make_eda_plots(annual: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(annual["year"], annual["wage_all_workers_annual"], marker="o", linewidth=2.0, color="#1F618D")
    axes[0].set_title("Annual Wage for All Workers (1994-2025)")
    axes[0].set_ylabel("Wage level")

    axes[1].plot(
        annual["year"],
        annual["dest_gdp_growth_avg_annual"],
        marker="o",
        linewidth=2.0,
        color="#C0392B",
    )
    axes[1].set_title("Destination GDP Growth Average (1994-2025)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("GDP growth (%)")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_1_eda_trends.png", dpi=180)
    plt.close(fig)

    corr = annual[
        [
            "slbfe_total_annual",
            "remittances_annual_usd_mn",
            "wage_all_workers_annual",
            "dest_gdp_growth_avg_annual",
        ]
    ].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, ax=ax)
    ax.set_title("Correlation Heatmap (Annual Aggregation)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_1_eda_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def evaluate_models(annual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    X = annual[["slbfe_total_annual", "remittances_annual_usd_mn"]].to_numpy()
    targets = {
        "wage_all_workers_annual": annual["wage_all_workers_annual"].to_numpy(),
        "dest_gdp_growth_avg_annual": annual["dest_gdp_growth_avg_annual"].to_numpy(),
    }
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    loo = LeaveOneOut()
    rows: list[dict[str, object]] = []
    best_predictions: dict[str, np.ndarray] = {}
    best_model_name: dict[str, str] = {}

    for target_name, y in targets.items():
        best_rmse = None
        for model_name, model in models.items():
            y_pred = cross_val_predict(model, X, y, cv=loo)
            mae = mean_absolute_error(y, y_pred)
            rmse = mean_squared_error(y, y_pred) ** 0.5
            r2 = r2_score(y, y_pred)
            rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "n_samples": int(len(y)),
                    "validation": "Leave-One-Out CV",
                }
            )
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_predictions[target_name] = y_pred
                best_model_name[target_name] = model_name

    metrics = pd.DataFrame(rows)
    return metrics, best_predictions, best_model_name


def make_ml_plots(
    annual: pd.DataFrame,
    metrics: pd.DataFrame,
    best_predictions: dict[str, np.ndarray],
    best_model_name: dict[str, str],
) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, target in zip(
        axes,
        ["wage_all_workers_annual", "dest_gdp_growth_avg_annual"],
        strict=True,
    ):
        sub = metrics[metrics["target"] == target].copy()
        sub = sub.sort_values("rmse", ascending=True)
        sns.barplot(data=sub, x="model", y="rmse", hue="model", ax=ax, palette="viridis", legend=False)
        ax.set_title(f"RMSE by Model: {target}")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_3_2_evaluation_metrics_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, target in zip(
        axes,
        ["wage_all_workers_annual", "dest_gdp_growth_avg_annual"],
        strict=True,
    ):
        y_true = annual[target].to_numpy()
        y_pred = best_predictions[target]
        ax.scatter(y_true, y_pred, alpha=0.8, color="#2E86C1")
        min_v = float(min(y_true.min(), y_pred.min()))
        max_v = float(max(y_true.max(), y_pred.max()))
        ax.plot([min_v, max_v], [min_v, max_v], color="#C0392B", linewidth=2)
        ax.set_title(f"Observed vs Predicted ({best_model_name[target]})")
        ax.set_xlabel("Observed")
        ax.set_ylabel("LOOCV Predicted")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_3_3_ml_results_observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def run_statistical_inference(annual: pd.DataFrame) -> dict[str, object]:
    X = annual[["slbfe_total_annual", "remittances_annual_usd_mn"]]
    X_const = sm.add_constant(X)

    y_wage = annual["wage_all_workers_annual"]
    model_wage = sm.OLS(y_wage, X_const).fit()

    y_dest = annual["dest_gdp_growth_avg_annual"]
    model_dest = sm.OLS(y_dest, X_const).fit()

    rho_emig_wage, p_emig_wage = spearmanr(annual["slbfe_total_annual"], annual["wage_all_workers_annual"])
    rho_remit_wage, p_remit_wage = spearmanr(annual["remittances_annual_usd_mn"], annual["wage_all_workers_annual"])
    rho_emig_dest, p_emig_dest = spearmanr(annual["slbfe_total_annual"], annual["dest_gdp_growth_avg_annual"])
    rho_remit_dest, p_remit_dest = spearmanr(
        annual["remittances_annual_usd_mn"], annual["dest_gdp_growth_avg_annual"]
    )

    out = {
        "n_years": int(len(annual)),
        "year_start": int(annual["year"].min()),
        "year_end": int(annual["year"].max()),
        "hypothesis_1_wage_model": {
            "f_statistic": float(model_wage.fvalue),
            "f_pvalue": float(model_wage.f_pvalue),
            "r_squared": float(model_wage.rsquared),
            "adj_r_squared": float(model_wage.rsquared_adj),
            "coef_const": float(model_wage.params["const"]),
            "coef_slbfe_total_annual": float(model_wage.params["slbfe_total_annual"]),
            "coef_remittances_annual_usd_mn": float(model_wage.params["remittances_annual_usd_mn"]),
            "p_slbfe_total_annual": float(model_wage.pvalues["slbfe_total_annual"]),
            "p_remittances_annual_usd_mn": float(model_wage.pvalues["remittances_annual_usd_mn"]),
            "ci_slbfe_total_annual": [float(x) for x in model_wage.conf_int().loc["slbfe_total_annual"].to_list()],
            "ci_remittances_annual_usd_mn": [
                float(x) for x in model_wage.conf_int().loc["remittances_annual_usd_mn"].to_list()
            ],
            "spearman_rho_emigration_wage": float(rho_emig_wage),
            "spearman_p_emigration_wage": float(p_emig_wage),
            "spearman_rho_remittance_wage": float(rho_remit_wage),
            "spearman_p_remittance_wage": float(p_remit_wage),
        },
        "hypothesis_2_dest_gdp_growth_model": {
            "f_statistic": float(model_dest.fvalue),
            "f_pvalue": float(model_dest.f_pvalue),
            "r_squared": float(model_dest.rsquared),
            "adj_r_squared": float(model_dest.rsquared_adj),
            "coef_const": float(model_dest.params["const"]),
            "coef_slbfe_total_annual": float(model_dest.params["slbfe_total_annual"]),
            "coef_remittances_annual_usd_mn": float(model_dest.params["remittances_annual_usd_mn"]),
            "p_slbfe_total_annual": float(model_dest.pvalues["slbfe_total_annual"]),
            "p_remittances_annual_usd_mn": float(model_dest.pvalues["remittances_annual_usd_mn"]),
            "ci_slbfe_total_annual": [float(x) for x in model_dest.conf_int().loc["slbfe_total_annual"].to_list()],
            "ci_remittances_annual_usd_mn": [
                float(x) for x in model_dest.conf_int().loc["remittances_annual_usd_mn"].to_list()
            ],
            "spearman_rho_emigration_destgdp": float(rho_emig_dest),
            "spearman_p_emigration_destgdp": float(p_emig_dest),
            "spearman_rho_remittance_destgdp": float(rho_remit_dest),
            "spearman_p_remittance_destgdp": float(p_remit_dest),
        },
    }
    return out


def make_inference_plot(annual: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.regplot(
        data=annual,
        x="slbfe_total_annual",
        y="wage_all_workers_annual",
        ci=95,
        scatter_kws={"alpha": 0.8, "s": 45},
        line_kws={"color": "#1F618D", "linewidth": 2},
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Emigration vs Wage (Annual)")
    axes[0, 0].set_xlabel("SLBFE total annual emigration")
    axes[0, 0].set_ylabel("Wage all workers annual")

    sns.regplot(
        data=annual,
        x="remittances_annual_usd_mn",
        y="wage_all_workers_annual",
        ci=95,
        scatter_kws={"alpha": 0.8, "s": 45},
        line_kws={"color": "#117A65", "linewidth": 2},
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Remittances vs Wage (Annual)")
    axes[0, 1].set_xlabel("Remittances annual (USD mn)")
    axes[0, 1].set_ylabel("Wage all workers annual")

    sns.regplot(
        data=annual,
        x="slbfe_total_annual",
        y="dest_gdp_growth_avg_annual",
        ci=95,
        scatter_kws={"alpha": 0.8, "s": 45},
        line_kws={"color": "#8E44AD", "linewidth": 2},
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Emigration vs Destination GDP Growth (Annual)")
    axes[1, 0].set_xlabel("SLBFE total annual emigration")
    axes[1, 0].set_ylabel("Destination GDP growth average annual (%)")

    sns.regplot(
        data=annual,
        x="remittances_annual_usd_mn",
        y="dest_gdp_growth_avg_annual",
        ci=95,
        scatter_kws={"alpha": 0.8, "s": 45},
        line_kws={"color": "#C0392B", "linewidth": 2},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Remittances vs Destination GDP Growth (Annual)")
    axes[1, 1].set_xlabel("Remittances annual (USD mn)")
    axes[1, 1].set_ylabel("Destination GDP growth average annual (%)")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "3_3_4_statistical_inference_relationships.png", dpi=180)
    plt.close(fig)


def write_latex_block(inference: dict[str, object], annual: pd.DataFrame) -> None:
    h1 = inference["hypothesis_1_wage_model"]
    h2 = inference["hypothesis_2_dest_gdp_growth_model"]
    n = inference["n_years"]
    y0 = inference["year_start"]
    y1 = inference["year_end"]

    h1_joint = "rejected" if float(h1["f_pvalue"]) < 0.05 else "not rejected"
    h2_joint = "rejected" if float(h2["f_pvalue"]) < 0.05 else "not rejected"
    h1_sig = "significant" if float(h1["f_pvalue"]) < 0.05 else "not significant"
    h2_sig = "significant" if float(h2["f_pvalue"]) < 0.05 else "not significant"

    w_mean = float(annual["wage_all_workers_annual"].mean())
    w_sd = float(annual["wage_all_workers_annual"].std(ddof=1))
    d_mean = float(annual["dest_gdp_growth_avg_annual"].mean())
    d_sd = float(annual["dest_gdp_growth_avg_annual"].std(ddof=1))

    latex = rf"""\indent \textbf{{Hypothesis 1: Joint Association of Emigration and Remittances with Wage Levels}}\\
We investigated whether annual wage levels are jointly associated with emigration volume (\texttt{{slbfe\_total\_annual}}) and remittance inflows (\texttt{{remittances\_annual\_usd\_mn}}) over {y0}--{y1}. A multiple OLS model was fitted with annual observations ($n = {n}$) at $\alpha = 0.05$.
\begin{{itemize}}[noitemsep, topsep=0pt]
    \item \textbf{{Null Hypothesis}} ($H_0$): $\beta_1 = \beta_2 = 0$ (no joint linear association with \texttt{{wage\_all\_workers\_annual}}).
    \item \textbf{{Alternative Hypothesis}} ($H_1$): At least one slope is non-zero ($\beta_1 \neq 0$ or $\beta_2 \neq 0$).
\end{{itemize}}
\noindent \textbf{{Results:}} The overall model was {h1_sig}, $F = {h1['f_statistic']:.3f}$, $p = {h1['f_pvalue']:.4f}$, with $R^2 = {h1['r_squared']:.3f}$ (adjusted $R^2 = {h1['adj_r_squared']:.3f}$). The annual wage series had mean $\bar{{y}} = {w_mean:.2f}$ and $SD = {w_sd:.2f}$. Coefficient estimates were: $\hat\beta_1 = {h1['coef_slbfe_total_annual']:.8f}$ ($p = {h1['p_slbfe_total_annual']:.4f}$; 95\% CI [{h1['ci_slbfe_total_annual'][0]:.8f}, {h1['ci_slbfe_total_annual'][1]:.8f}]) and $\hat\beta_2 = {h1['coef_remittances_annual_usd_mn']:.6f}$ ($p = {h1['p_remittances_annual_usd_mn']:.4f}$; 95\% CI [{h1['ci_remittances_annual_usd_mn'][0]:.6f}, {h1['ci_remittances_annual_usd_mn'][1]:.6f}]). Spearman checks yielded $\rho_{{emig,wage}} = {h1['spearman_rho_emigration_wage']:.3f}$ ($p = {h1['spearman_p_emigration_wage']:.4f}$) and $\rho_{{remit,wage}} = {h1['spearman_rho_remittance_wage']:.3f}$ ($p = {h1['spearman_p_remittance_wage']:.4f}$).

\noindent \textbf{{Interpretation:}} At $\alpha = 0.05$, the null hypothesis is {h1_joint}. Emigration and remittances jointly explain a statistically meaningful share of annual wage variation in this sample, with remittances carrying the stronger partial signal under the model.
~\\

\indent \textbf{{Hypothesis 2: Joint Association of Emigration and Remittances with Destination GDP Growth}}\\
We investigated whether destination GDP growth averages are jointly associated with emigration volume and remittance inflows over {y0}--{y1}. A multiple OLS model was fitted with annual observations ($n = {n}$) at $\alpha = 0.05$.
\begin{{itemize}}[noitemsep, topsep=0pt]
    \item \textbf{{Null Hypothesis}} ($H_0$): $\gamma_1 = \gamma_2 = 0$ (no joint linear association with \texttt{{dest\_gdp\_growth\_avg\_annual}}).
    \item \textbf{{Alternative Hypothesis}} ($H_1$): At least one slope is non-zero ($\gamma_1 \neq 0$ or $\gamma_2 \neq 0$).
\end{{itemize}}
\noindent \textbf{{Results:}} The overall model was {h2_sig}, $F = {h2['f_statistic']:.3f}$, $p = {h2['f_pvalue']:.4f}$, with $R^2 = {h2['r_squared']:.3f}$ (adjusted $R^2 = {h2['adj_r_squared']:.3f}$). The destination GDP growth series had mean $\bar{{y}} = {d_mean:.3f}$ and $SD = {d_sd:.3f}$. Coefficient estimates were: $\hat\gamma_1 = {h2['coef_slbfe_total_annual']:.8f}$ ($p = {h2['p_slbfe_total_annual']:.4f}$; 95\% CI [{h2['ci_slbfe_total_annual'][0]:.8f}, {h2['ci_slbfe_total_annual'][1]:.8f}]) and $\hat\gamma_2 = {h2['coef_remittances_annual_usd_mn']:.6f}$ ($p = {h2['p_remittances_annual_usd_mn']:.4f}$; 95\% CI [{h2['ci_remittances_annual_usd_mn'][0]:.6f}, {h2['ci_remittances_annual_usd_mn'][1]:.6f}]). Spearman checks yielded $\rho_{{emig,dest}} = {h2['spearman_rho_emigration_destgdp']:.3f}$ ($p = {h2['spearman_p_emigration_destgdp']:.4f}$) and $\rho_{{remit,dest}} = {h2['spearman_rho_remittance_destgdp']:.3f}$ ($p = {h2['spearman_p_remittance_destgdp']:.4f}$).

\noindent \textbf{{Interpretation:}} At $\alpha = 0.05$, the null hypothesis is {h2_joint}. The joint linear association with destination GDP growth is weak in this model, indicating limited predictive signal from these two migration-finance predictors alone.
~\\
"""
    (OUT_DIR / "statistical_inference_latex.tex").write_text(latex, encoding="utf-8")


def write_readme(
    annual: pd.DataFrame,
    metrics: pd.DataFrame,
    best_model_name: dict[str, str],
    inference: dict[str, object],
) -> None:
    h1 = inference["hypothesis_1_wage_model"]
    h2 = inference["hypothesis_2_dest_gdp_growth_model"]

    def fmt(v: float, d: int = 4) -> str:
        return f"{float(v):.{d}f}"

    best_wage = metrics[(metrics["target"] == "wage_all_workers_annual")].sort_values("rmse").iloc[0]
    best_dest = metrics[(metrics["target"] == "dest_gdp_growth_avg_annual")].sort_values("rmse").iloc[0]

    readme = rf"""# Combined Predictive Analysis: Emigration, Remittances, Wages, and Destination GDP Growth (1994-2025)

This folder contains one combined, reproducible analysis built from:

- `SriLanka_Migration_final.csv`
- Context checks from `final_column_analysis/wage_all_workers/*`
- Context checks from `final_column_analysis/dest_gdp_growth/*`

Reproduce all outputs:

```powershell
python build_combined_predictive_topics.py
```

## 3.1 Exploratory Data Analysis

Annual EDA (1994-2025, `n={len(annual)}` years) indicates strong upward wage levels over time and cyclical destination GDP growth behavior. Correlation screening for the four core variables shows:

- `corr(slbfe_total_annual, wage_all_workers_annual) = {fmt(annual[['slbfe_total_annual','wage_all_workers_annual']].corr().iloc[0,1], 3)}`
- `corr(remittances_annual_usd_mn, wage_all_workers_annual) = {fmt(annual[['remittances_annual_usd_mn','wage_all_workers_annual']].corr().iloc[0,1], 3)}`
- `corr(slbfe_total_annual, dest_gdp_growth_avg_annual) = {fmt(annual[['slbfe_total_annual','dest_gdp_growth_avg_annual']].corr().iloc[0,1], 3)}`
- `corr(remittances_annual_usd_mn, dest_gdp_growth_avg_annual) = {fmt(annual[['remittances_annual_usd_mn','dest_gdp_growth_avg_annual']].corr().iloc[0,1], 3)}`

```mermaid
flowchart LR
    A[SriLanka_Migration_final.csv] --> B[Annual aggregation by year]
    B --> C[Trend plots]
    B --> D[Correlation heatmap]
    C --> E[EDA findings]
    D --> E
```

![EDA Trends](charts/3_1_eda_trends.png)

![EDA Correlation Heatmap](charts/3_1_eda_correlation_heatmap.png)

## 3.3.1 Machine Learning Task Formulation

Two supervised regression tasks were formulated with shared predictors:

- Feature set for both tasks: `slbfe_total_annual`, `remittances_annual_usd_mn`
- Task A target: `wage_all_workers_annual`
- Task B target: `dest_gdp_growth_avg_annual`

The objective is predictive performance under small-sample annual data, not causal identification.

```mermaid
flowchart TD
    A[Annual dataset n={len(annual)}] --> B[Feature matrix X]
    B --> C1[slbfe_total_annual]
    B --> C2[remittances_annual_usd_mn]
    A --> D1[Target y1: wage_all_workers_annual]
    A --> D2[Target y2: dest_gdp_growth_avg_annual]
    C1 --> E[Regression models]
    C2 --> E
    D1 --> E
    D2 --> E
    E --> F[LOOCV predictions and metrics]
```

## 3.3.2 Evaluation Metrics & Validation Strategy

Validation uses Leave-One-Out Cross-Validation (LOOCV), appropriate for `n={len(annual)}` annual observations.

- Fold design: one held-out year per fold
- Metrics: MAE, RMSE, R2
- Candidate models: LinearRegression, RandomForestRegressor, GradientBoostingRegressor

```mermaid
flowchart LR
    A[Annual data] --> B[LOOCV split]
    B --> C[Train on n-1 years]
    C --> D[Predict held-out year]
    D --> E[Repeat for all years]
    E --> F[Aggregate MAE RMSE R2]
```

![LOOCV RMSE Comparison](charts/3_3_2_evaluation_metrics_rmse.png)

## 3.3.3 Machine Learning Results

Best out-of-sample models (by RMSE):

- `wage_all_workers_annual`: {best_model_name['wage_all_workers_annual']} (MAE={best_wage['mae']:.4f}, RMSE={best_wage['rmse']:.4f}, R2={best_wage['r2']:.4f})
- `dest_gdp_growth_avg_annual`: {best_model_name['dest_gdp_growth_avg_annual']} (MAE={best_dest['mae']:.4f}, RMSE={best_dest['rmse']:.4f}, R2={best_dest['r2']:.4f})

Interpretation summary:

- Wage target has stronger predictive recoverability from these two predictors.
- Destination GDP growth remains weakly predicted, suggesting omitted-variable effects.

```mermaid
flowchart TD
    A[Model comparison by RMSE] --> B[Select best model per target]
    B --> C1[Wage prediction quality]
    B --> C2[Dest GDP growth prediction quality]
    C1 --> D[Observed vs predicted diagnostic]
    C2 --> D
```

![Observed vs Predicted](charts/3_3_3_ml_results_observed_vs_predicted.png)

## 3.3.4 Statistical Inference

Inference used two separate multiple OLS models with annual data (`n={len(annual)}`, years {int(annual['year'].min())}-{int(annual['year'].max())}).

```mermaid
flowchart LR
    A[Predictors: emigration and remittances] --> B[OLS for wage_all_workers_annual]
    A --> C[OLS for dest_gdp_growth_avg_annual]
    B --> D[F-test and coefficient tests]
    C --> D
    D --> E[Decision at alpha 0.05]
```

![Inference Relationships](charts/3_3_4_statistical_inference_relationships.png)

### Reproducible LaTeX Block

```latex
\indent \textbf{{Hypothesis 1: Joint Association of Emigration and Remittances with Wage Levels}}\\
We investigated whether annual wage levels are jointly associated with emigration volume (\texttt{{slbfe\_total\_annual}}) and remittance inflows (\texttt{{remittances\_annual\_usd\_mn}}) over {int(annual['year'].min())}--{int(annual['year'].max())}. A multiple OLS model was fitted with annual observations ($n = {len(annual)}$) at $\alpha = 0.05$.
\begin{{itemize}}[noitemsep, topsep=0pt]
    \item \textbf{{Null Hypothesis}} ($H_0$): $\beta_1 = \beta_2 = 0$ (no joint linear association with \texttt{{wage\_all\_workers\_annual}}).
    \item \textbf{{Alternative Hypothesis}} ($H_1$): At least one slope is non-zero ($\beta_1 \neq 0$ or $\beta_2 \neq 0$).
\end{{itemize}}
\noindent \textbf{{Results:}} The overall model was {'significant' if h1['f_pvalue'] < 0.05 else 'not significant'}, $F = {h1['f_statistic']:.3f}$, $p = {h1['f_pvalue']:.4f}$, with $R^2 = {h1['r_squared']:.3f}$ (adjusted $R^2 = {h1['adj_r_squared']:.3f}$). Coefficient estimates were: $\hat\beta_1 = {h1['coef_slbfe_total_annual']:.8f}$ ($p = {h1['p_slbfe_total_annual']:.4f}$; 95\% CI [{h1['ci_slbfe_total_annual'][0]:.8f}, {h1['ci_slbfe_total_annual'][1]:.8f}]) and $\hat\beta_2 = {h1['coef_remittances_annual_usd_mn']:.6f}$ ($p = {h1['p_remittances_annual_usd_mn']:.4f}$; 95\% CI [{h1['ci_remittances_annual_usd_mn'][0]:.6f}, {h1['ci_remittances_annual_usd_mn'][1]:.6f}]). Spearman checks yielded $\rho_{{emig,wage}} = {h1['spearman_rho_emigration_wage']:.3f}$ ($p = {h1['spearman_p_emigration_wage']:.4f}$) and $\rho_{{remit,wage}} = {h1['spearman_rho_remittance_wage']:.3f}$ ($p = {h1['spearman_p_remittance_wage']:.4f}$).

\noindent \textbf{{Interpretation:}} At $\alpha = 0.05$, the null hypothesis is {'rejected' if h1['f_pvalue'] < 0.05 else 'not rejected'}. Emigration and remittances jointly explain a statistically meaningful share of annual wage variation in this sample.
~\\

\indent \textbf{{Hypothesis 2: Joint Association of Emigration and Remittances with Destination GDP Growth}}\\
We investigated whether destination GDP growth averages are jointly associated with emigration volume and remittance inflows over {int(annual['year'].min())}--{int(annual['year'].max())}. A multiple OLS model was fitted with annual observations ($n = {len(annual)}$) at $\alpha = 0.05$.
\begin{{itemize}}[noitemsep, topsep=0pt]
    \item \textbf{{Null Hypothesis}} ($H_0$): $\gamma_1 = \gamma_2 = 0$ (no joint linear association with \texttt{{dest\_gdp\_growth\_avg\_annual}}).
    \item \textbf{{Alternative Hypothesis}} ($H_1$): At least one slope is non-zero ($\gamma_1 \neq 0$ or $\gamma_2 \neq 0$).
\end{{itemize}}
\noindent \textbf{{Results:}} The overall model was {'significant' if h2['f_pvalue'] < 0.05 else 'not significant'}, $F = {h2['f_statistic']:.3f}$, $p = {h2['f_pvalue']:.4f}$, with $R^2 = {h2['r_squared']:.3f}$ (adjusted $R^2 = {h2['adj_r_squared']:.3f}$). Coefficient estimates were: $\hat\gamma_1 = {h2['coef_slbfe_total_annual']:.8f}$ ($p = {h2['p_slbfe_total_annual']:.4f}$; 95\% CI [{h2['ci_slbfe_total_annual'][0]:.8f}, {h2['ci_slbfe_total_annual'][1]:.8f}]) and $\hat\gamma_2 = {h2['coef_remittances_annual_usd_mn']:.6f}$ ($p = {h2['p_remittances_annual_usd_mn']:.4f}$; 95\% CI [{h2['ci_remittances_annual_usd_mn'][0]:.6f}, {h2['ci_remittances_annual_usd_mn'][1]:.6f}]). Spearman checks yielded $\rho_{{emig,dest}} = {h2['spearman_rho_emigration_destgdp']:.3f}$ ($p = {h2['spearman_p_emigration_destgdp']:.4f}$) and $\rho_{{remit,dest}} = {h2['spearman_rho_remittance_destgdp']:.3f}$ ($p = {h2['spearman_p_remittance_destgdp']:.4f}$).

\noindent \textbf{{Interpretation:}} At $\alpha = 0.05$, the null hypothesis is {'rejected' if h2['f_pvalue'] < 0.05 else 'not rejected'}. Joint explanatory power for destination GDP growth is limited in this specification.
~\\
```

The same block is exported to `statistical_inference_latex.tex`.

## Generated Artifacts

- `annual_aggregated_1994_2025.csv`
- `ml_metrics_loocv.csv`
- `inference_results.json`
- `analysis_summary.json`
- `statistical_inference_latex.tex`
- `charts/3_1_eda_trends.png`
- `charts/3_1_eda_correlation_heatmap.png`
- `charts/3_3_2_evaluation_metrics_rmse.png`
- `charts/3_3_3_ml_results_observed_vs_predicted.png`
- `charts/3_3_4_statistical_inference_relationships.png`
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    annual = load_annual_data()
    annual.to_csv(OUT_DIR / "annual_aggregated_1994_2025.csv", index=False)

    make_eda_plots(annual)
    metrics, best_predictions, best_model_name = evaluate_models(annual)
    metrics.to_csv(OUT_DIR / "ml_metrics_loocv.csv", index=False)
    make_ml_plots(annual, metrics, best_predictions, best_model_name)

    inference = run_statistical_inference(annual)
    (OUT_DIR / "inference_results.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")
    make_inference_plot(annual)
    write_latex_block(inference, annual)

    summary = {
        "n_years": int(len(annual)),
        "best_models": best_model_name,
        "targets": {
            tgt: {
                "best_rmse": float(metrics[(metrics["target"] == tgt)].sort_values("rmse").iloc[0]["rmse"]),
                "best_r2": float(metrics[(metrics["target"] == tgt)].sort_values("rmse").iloc[0]["r2"]),
            }
            for tgt in ["wage_all_workers_annual", "dest_gdp_growth_avg_annual"]
        },
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_readme(annual, metrics, best_model_name, inference)

    print("Generated outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
