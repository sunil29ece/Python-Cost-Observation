# Python Cost Observation — Monthly Comparison + ML

A GitHub-ready Python project to compare the latest two months of spending, describe *observed* cost drivers, flag unusual months with Isolation Forest, and suggest where to investigate savings.

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py
```

The project includes ten months of **synthetic example data** in `costs.csv`. Replace it with your own data (CAD). Each row is one month/category, with columns `Month,Category,Cost,Usage,UnitPrice,Notes`. `Cost` may include additional charges beyond `Usage * UnitPrice`; these charges need separate invoice investigation.

## Outputs

The `output/` folder contains:
- `monthly_comparison.csv` — previous/current cost, difference, percent change and possible explanation
- `ml_anomalies.csv` — month-level anomaly scores and flags
- `cost_recommendations.csv` — investigation actions and illustrative savings targets
- `monthly_trends.png` and `monthly_difference.png` — charts

## How it works

1. Compares the latest two dates in the input, not necessarily the calendar month today.
2. Attributes a cost increase to higher recorded usage or unit price when the values support it. Notes are user-entered; they are not independent proof.
3. Isolation Forest looks for unusual **monthly patterns** after at least six months of history. An anomaly is not automatically overspending, and small datasets can give unreliable flags.
4. Estimates a **non-guaranteed illustrative savings target** as the smaller of the amount above the historical category median or 10% of the current category cost. These estimates are heuristics, not ML predictions. Savings depend on which charges are actually avoidable.

## GitHub upload

Create a repository named `Python-Cost-Observation` and upload `main.py`, `costs.csv`, `requirements.txt`, `README.md`, `.gitignore`.

## Limitations

- Synthetic sample data is for demonstration only.
- The model does not infer causal explanations or predict actual savings.
- Rent and other fixed/essential costs may not be reducible.
- For more trustworthy ML, collect 12+ months and account for seasonality, business changes, and billing cycles.
