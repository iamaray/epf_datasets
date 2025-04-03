import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional
import warnings
import os
import glob
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the time series data."""
    df = pd.read_csv(file_path)

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def calculate_correlation_metrics(df: pd.DataFrame, target_col: str,
                                  covariate_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate various correlation metrics between target and covariates."""
    if covariate_cols is None:
        covariate_cols = [col for col in df.columns if col != target_col]

    results = []
    for col in covariate_cols:
        pearson_corr = df[target_col].corr(df[col])
        spearman_corr = df[target_col].corr(df[col], method='spearman')
        kendall_tau = df[target_col].corr(df[col], method='kendall')

        max_lag = 24
        cross_corrs = []
        for lag in range(-max_lag, max_lag + 1):
            cross_corr = df[target_col].corr(df[col].shift(lag))
            cross_corrs.append(cross_corr)
        max_cross_corr = max(cross_corrs)
        best_lag = cross_corrs.index(max_cross_corr) - max_lag

        results.append({
            'feature': col,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'kendall_tau': kendall_tau,
            'max_cross_correlation': max_cross_corr,
            'best_lag': best_lag
        })

    return pd.DataFrame(results)


def calculate_mutual_information(df: pd.DataFrame, target_col: str,
                                 covariate_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate mutual information between target and covariates."""
    if covariate_cols is None:
        covariate_cols = [col for col in df.columns if col != target_col]

    X = df[covariate_cols].fillna(method='ffill').fillna(method='bfill')
    y = df[target_col].fillna(method='ffill').fillna(method='bfill')

    mi_scores = mutual_info_regression(X, y)

    return pd.DataFrame({
        'feature': covariate_cols,
        'mutual_information': mi_scores
    })


def calculate_granger_causality(df: pd.DataFrame, target_col: str,
                                covariate_cols: Optional[List[str]] = None,
                                max_lag: int = 24) -> pd.DataFrame:
    """Calculate Granger causality between target and covariates."""
    if covariate_cols is None:
        covariate_cols = [col for col in df.columns if col != target_col]

    results = []
    for col in covariate_cols:
        data = df[[target_col, col]].dropna()

        try:
            test_result = grangercausalitytests(data, max_lag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_ftest'][1]
                        for i in range(max_lag)]
            min_p_value = min(p_values)
            best_lag = p_values.index(min_p_value) + 1
        except:
            min_p_value = np.nan
            best_lag = np.nan

        results.append({
            'feature': col,
            'granger_p_value': min_p_value,
            'granger_best_lag': best_lag
        })

    return pd.DataFrame(results)


def calculate_stationarity(df: pd.DataFrame, target_col: str,
                           covariate_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate stationarity metrics for each feature."""
    if covariate_cols is None:
        covariate_cols = [col for col in df.columns if col != target_col]

    results = []
    for col in covariate_cols:
        if df[col].nunique() == 1:
            results.append({
                'feature': col,
                'adf_statistic': np.nan,
                'adf_p_value': np.nan,
                'is_stationary': True,
                'is_constant': True
            })
            continue

        if df[col].dropna().empty:
            results.append({
                'feature': col,
                'adf_statistic': np.nan,
                'adf_p_value': np.nan,
                'is_stationary': np.nan,
                'is_constant': False
            })
            continue

        try:
            adf_result = adfuller(df[col].dropna())

            results.append({
                'feature': col,
                'adf_statistic': adf_result[0],
                'adf_p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'is_constant': False
            })
        except Exception as e:
            results.append({
                'feature': col,
                'adf_statistic': np.nan,
                'adf_p_value': np.nan,
                'is_stationary': np.nan,
                'is_constant': False
            })

    return pd.DataFrame(results)


def analyze_features(file_path: str, target_col: str,
                     covariate_cols: Optional[List[str]] = None,
                     output_path: Optional[str] = None) -> dict:
    """
    Main function to analyze feature importance for time series data.

    Args:
        file_path: Path to the CSV file
        target_col: Name of the target column
        covariate_cols: List of covariate columns to analyze (if None, use all except target)
        output_path: Optional path to save results as CSV

    Returns:
        Dictionary containing all analysis results
    """
    df = load_data(file_path)

    correlation_metrics = calculate_correlation_metrics(
        df, target_col, covariate_cols)
    mi_metrics = calculate_mutual_information(df, target_col, covariate_cols)
    granger_metrics = calculate_granger_causality(
        df, target_col, covariate_cols)
    stationarity_metrics = calculate_stationarity(
        df, target_col, covariate_cols)

    results = correlation_metrics.merge(mi_metrics, on='feature')
    results = results.merge(granger_metrics, on='feature')
    results = results.merge(stationarity_metrics, on='feature')

    results = results.sort_values('mutual_information', ascending=False)

    if output_path:
        results.to_csv(output_path, index=False)

    return {
        'results': results,
        'correlation_metrics': correlation_metrics,
        'mutual_information': mi_metrics,
        'granger_causality': granger_metrics,
        'stationarity': stationarity_metrics
    }


def create_latex_table(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Create a LaTeX table of the top 10 features based on mutual information.

    Args:
        results_df: DataFrame containing the analysis results
        output_path: Path to save the LaTeX table
    """
    top_features = results_df.nlargest(10, 'mutual_information')

    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Top 10 Features by Mutual Information}\n"
    latex_table += "\\label{tab:top_features}\n"
    latex_table += "\\begin{tabular}{lcc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Feature & Mutual Information & Pearson Correlation \\\\\n"
    latex_table += "\\midrule\n"

    for _, row in top_features.iterrows():
        latex_table += f"{row['feature']} & {row['mutual_information']:.4f} & {row['pearson_correlation']:.4f} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    with open(output_path, 'w') as f:
        f.write(latex_table)


if __name__ == "__main__":
    # Get all CSV files in the cleaned data directory
    data_dir = "data/cleaned"  # Changed from "../data/cleaned"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
    else:
        print(f"Found {len(csv_files)} CSV files to process")

        for file_path in csv_files:
            # Extract the base name without extension for output files
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"\nProcessing {base_name}...")

            # Set target column based on file name
            if "spain" in base_name.lower():
                target_col = "total load actual"
            elif "homestead" in base_name.lower():
                target_col = "Consumption"
            elif "ercot" in base_name.lower():
                target_col = "ACTUAL_NetLoad"

            results = analyze_features(
                file_path=file_path,
                target_col=target_col,
                covariate_cols=None,
                output_path=f"{base_name}_feature_importance.csv"
            )

            # Print top 10 most important features
            print(f"\nTop 10 most important features for {base_name}:")
            print(results['results'].head(10))

            # Save the full results DataFrame
            results['results'].to_csv(
                f"{base_name}_feature_importance.csv", index=False)
            print(f"Results saved to '{base_name}_feature_importance.csv'")

            # Create and save LaTeX table
            create_latex_table(results['results'],
                               f"{base_name}_top_features_latex.tex")
            print(f"LaTeX table saved to '{base_name}_top_features_latex.tex'")
