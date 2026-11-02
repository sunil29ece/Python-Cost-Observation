"""Monthly cost observability, explainable variance and ML anomaly detection."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'costs.csv'
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(exist_ok=True)


def load_costs(path=INPUT):
    df = pd.read_csv(path, parse_dates=['Month'])
    required = {'Month', 'Category', 'Cost', 'Usage', 'UnitPrice', 'Notes'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    if df[['Cost', 'Usage', 'UnitPrice']].isna().any().any() or (df[['Cost', 'Usage', 'UnitPrice']] < 0).any().any():
        raise ValueError('Costs, usage and unit prices must be nonnegative numbers.')
    if df.duplicated(['Month', 'Category']).any():
        raise ValueError('One row per month/category is required.')
    return df.sort_values(['Month', 'Category']).reset_index(drop=True)


def compare_months(df):
    months = sorted(df.Month.unique())
    if len(months) < 2:
        raise ValueError('At least two months of data are required.')
    previous, current = months[-2], months[-1]
    prev = df[df.Month == previous].set_index('Category')
    curr = df[df.Month == current].set_index('Category')
    categories = sorted(set(prev.index) | set(curr.index))
    rows = []
    for category in categories:
        old = float(prev.loc[category, 'Cost']) if category in prev.index else 0.0
        new = float(curr.loc[category, 'Cost']) if category in curr.index else 0.0
        delta = new - old
        pct = (delta / old * 100) if old else None
        if category not in prev.index:
            explanation = 'New category this month; investigate the new expense.'
        elif category not in curr.index:
            explanation = 'Category absent this month; verify whether spending stopped.'
        else:
            old_usage, new_usage = float(prev.loc[category, 'Usage']), float(curr.loc[category, 'Usage'])
            old_price, new_price = float(prev.loc[category, 'UnitPrice']), float(curr.loc[category, 'UnitPrice'])
            if delta <= 0:
                explanation = 'No cost increase.'
            else:
                drivers = []
                if new_usage > old_usage:
                    drivers.append(f'usage rose {new_usage-old_usage:.1f} units')
                if new_price > old_price:
                    drivers.append(f'unit price rose ${new_price-old_price:.2f}')
                if drivers:
                    explanation = 'Observed drivers: ' + ' and '.join(drivers) + '.'
                else:
                    explanation = 'Cost increased without higher recorded usage or unit price; review fees or data.'
                note = str(curr.loc[category, 'Notes']).strip()
                if note and note.lower() != 'nan':
                    explanation += f' Recorded note: {note}.'
        rows.append({'Category': category, 'PreviousCost': old, 'CurrentCost': new,
                     'Difference': delta, 'ChangePct': pct, 'Explanation': explanation})
    result = pd.DataFrame(rows).sort_values('Difference', ascending=False)
    return pd.Timestamp(previous), pd.Timestamp(current), result


def detect_anomalies(df):
    """Train on monthly category spending; small histories are not reliable for ML."""
    pivot = df.pivot(index='Month', columns='Category', values='Cost').fillna(0).sort_index()
    if len(pivot) < 6:
        return pd.DataFrame(columns=['Month', 'AnomalyScore', 'Anomaly']), 'Not enough history for reliable ML (need >= 6 months).'
    model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
    pred = model.fit_predict(pivot)
    score = model.decision_function(pivot)
    result = pd.DataFrame({'Month': pivot.index, 'AnomalyScore': score, 'Anomaly': pred == -1})
    return result, 'Isolation Forest flags unusual monthly spending patterns, not proven waste or causes.'


def recommendations(comparison, df):
    history = df.groupby('Category')['Cost'].median()
    rows = []
    for _, row in comparison.iterrows():
        if row.Difference <= 0:
            continue
        category = row.Category
        median = float(history.get(category, 0))
        # Illustrative, capped target: only excess above historic median, at most 10% of current spending.
        target = min(max(row.CurrentCost - median, 0), row.CurrentCost * 0.10)
        suggestions = {
            'Food': 'Review dining-out and grocery purchases; compare unit prices and reduce waste.',
            'Transport': 'Review trip frequency, fuel usage, transit passes and parking fees.',
            'Rent': 'Check rent adjustments and recurring housing charges; avoid assuming rent is discretionary.',
            'Utilities': 'Check metered usage, seasonal effects, rates and fixed charges.',
            'Shopping': 'Review nonessential purchases, returns and subscriptions.',
        }
        rows.append({'Category': category, 'Increase': row.Difference,
                     'IllustrativeSavingsTarget': round(target, 2),
                     'Action': suggestions.get(category, 'Review invoices, quantity, rates and recurring charges.')})
    return pd.DataFrame(rows, columns=['Category', 'Increase', 'IllustrativeSavingsTarget', 'Action'])


def charts(df, comparison):
    pivot = df.pivot(index='Month', columns='Category', values='Cost').fillna(0).sort_index()
    ax = pivot.plot(figsize=(11, 5), marker='o')
    ax.set(title='Monthly cost trends', xlabel='Month', ylabel='Cost (CAD)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT / 'monthly_trends.png', dpi=150)
    plt.close()
    ax = comparison.set_index('Category')['Difference'].sort_values().plot.barh(figsize=(9, 5))
    ax.set(title='Current minus previous month', xlabel='Cost difference (CAD)')
    plt.tight_layout()
    plt.savefig(OUTPUT / 'monthly_difference.png', dpi=150)
    plt.close()


def main():
    df = load_costs()
    prev, curr, comparison = compare_months(df)
    anomalies, ml_note = detect_anomalies(df)
    savings = recommendations(comparison, df)
    comparison.to_csv(OUTPUT / 'monthly_comparison.csv', index=False)
    anomalies.to_csv(OUTPUT / 'ml_anomalies.csv', index=False)
    savings.to_csv(OUTPUT / 'cost_recommendations.csv', index=False)
    charts(df, comparison)
    old_total, new_total = comparison.PreviousCost.sum(), comparison.CurrentCost.sum()
    print(f'Previous month: {prev:%Y-%m} = CAD ${old_total:,.2f}')
    print(f'Current month:  {curr:%Y-%m} = CAD ${new_total:,.2f}')
    print(f'Change: CAD ${new_total-old_total:+,.2f} ({(new_total-old_total)/old_total*100:+.1f}%)' if old_total else 'Percentage change unavailable (previous total was zero).')
    print('\nCategory variance and evidence-based explanations:')
    print(comparison.to_string(index=False))
    print('\nML:', ml_note)
    print(anomalies.to_string(index=False))
    print('\nOptimization opportunities (illustrative targets, NOT guaranteed savings):')
    print(savings.to_string(index=False))
    print(f'\nReports and charts saved to: {OUTPUT}')


if __name__ == '__main__':
    main()
