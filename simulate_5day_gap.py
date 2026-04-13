import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')
from deployment_gap_recovery_extended import ExtendedDeploymentAlgorithm

# Load the data
df = pd.read_csv('data/2023_2026_merged.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define the 5-day gap window (choose a recent, full week with enough history)
gap_start = pd.Timestamp('2026-02-10 00:00:00')  # Tuesday
gap_end = pd.Timestamp('2026-02-14 23:50:00')    # Saturday

sites = ['Ptot_HA', 'Ptot_HEI', 'Ptot_HEI_13RT', 'Ptot_HEI_5RNS', 'Ptot_RIZOMM']

# Store actual values before creating gap
actual_values = {}
for site in sites:
    actual_values[site] = df.loc[gap_start:gap_end, site].copy()

# Inject the 5-day gap (set to NaN)
df_with_gap = df.copy()
df_with_gap.loc[gap_start:gap_end, sites] = np.nan

# Prepare data for algorithm
df_impute = df_with_gap.reset_index()[['Date'] + sites].copy()
df_impute.rename(columns={'Date': 'Timestamp'}, inplace=True)

# Run the latest algorithm
algorithm = ExtendedDeploymentAlgorithm(
    site_cols=sites,
    use_multi_week_templates=True,
    use_chunked_recovery=True,
    template_lookback_days=28,
    use_smart_chunking=True,
    adaptive_template_bias=0.70
)

print(f"\nSimulating 5-day gap: {gap_start.date()} to {gap_end.date()}\n")
df_recovered = algorithm.impute(df_impute)

# Extract recovered data
recovered_data = {}
for site in sites:
    recovered_data[site] = df_recovered[['Timestamp', site]].copy()
    recovered_data[site].set_index('Timestamp', inplace=True)

# Calculate and print metrics
print("Results (5-day gap):")
for site in sites:
    actual = actual_values[site]
    recovered = recovered_data[site].loc[gap_start:gap_end, site]
    mae = np.abs(actual.values - recovered.values).mean() / 1000
    mape = (np.abs((actual.values - recovered.values) / actual.values) * 100).mean()
    rmse = np.sqrt(np.mean((actual.values - recovered.values)**2)) / 1000
    r2 = 1 - (np.sum((actual.values - recovered.values)**2) / np.sum((actual.values - actual.mean())**2))
    print(f"{site}: MAE={mae:.2f} kW, MAPE={mape:.1f}%, RMSE={rmse:.2f} kW, R²={r2:.3f}")

# Visualize actual vs recovered for the gap
for site in sites:
    fig, ax = plt.subplots(figsize=(14, 6))
    actual = actual_values[site]
    recovered = recovered_data[site].loc[gap_start:gap_end, site]
    ax.plot(actual.index, actual.values / 1000, 'b-', label='Actual', linewidth=2.5)
    ax.plot(recovered.index, recovered.values / 1000, 'g--', label='Recovered', linewidth=2)
    ax.fill_between(actual.index, actual.values / 1000, recovered.values / 1000, color='orange', alpha=0.2, label='Error')
    ax.set_title(f'{site}: 5-Day Gap Recovery ({gap_start.date()} to {gap_end.date()})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Power (kW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'gap_4day/gap_enhanced_v2_5day_gap_{site}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fname}")
    plt.close()
