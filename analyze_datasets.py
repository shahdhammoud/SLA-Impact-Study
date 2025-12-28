import pandas as pd
import os

csv_dir = 'benchmarks_with_ground_truth/csv'
datasets = sorted([f.replace('.csv', '') for f in os.listdir(csv_dir) if f.endswith('.csv')])

print("=" * 90)
print("DATASET ANALYSIS FOR PLANNING")
print("=" * 90)
print(f"{'Dataset':<20} {'Rows':<10} {'Cols':<8} {'Category':<12} {'Data Types'}")
print("-" * 90)

small = []
medium = []
large = []

for d in datasets:
    try:
        df = pd.read_csv(f'{csv_dir}/{d}.csv')
        rows = len(df)
        cols = len(df.columns)

        dtypes = df.dtypes.value_counts().to_dict()
        dtype_str = ', '.join([f"{str(k).split('.')[-1]}:{v}" for k, v in dtypes.items()])

        if cols <= 10:
            category = "Small"
            small.append((d, rows, cols))
        elif cols <= 30:
            category = "Medium"
            medium.append((d, rows, cols))
        else:
            category = "Large"
            large.append((d, rows, cols))

        print(f"{d:<20} {rows:<10} {cols:<8} {category:<12} {dtype_str}")
    except Exception as e:
        print(f"{d:<20} ERROR: {e}")

print("\n" + "=" * 90)
print("RECOMMENDED ORDER (Small → Large):")
print("=" * 90)
print("\n🟢 SMALL DATASETS (≤10 features) - Start here:")
for d, r, c in sorted(small, key=lambda x: x[2]):
    print(f"   {d}: {c} features, {r} rows")

print("\n🟡 MEDIUM DATASETS (11-30 features):")
for d, r, c in sorted(medium, key=lambda x: x[2]):
    print(f"   {d}: {c} features, {r} rows")

print("\n🔴 LARGE DATASETS (>30 features) - Most complex:")
for d, r, c in sorted(large, key=lambda x: x[2]):
    print(f"   {d}: {c} features, {r} rows")
