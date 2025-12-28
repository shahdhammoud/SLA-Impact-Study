import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def identify_variable_types(df, categorical_features=None):
    if categorical_features is not None:
        cat_cols = categorical_features
        num_cols = [col for col in df.columns if col not in cat_cols]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [col for col in df.columns if col not in num_cols]
    return num_cols, cat_cols

def preprocess_for_ci(df, cat_cols):
    if cat_cols:
        return pd.get_dummies(df, columns=cat_cols), True
    return df.copy(), False

def load_ground_truth_graph(graph_path, columns):
    edges = []
    with open(graph_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                src, tgt = parts[0], parts[1]
                edges.append((src, tgt))
    n = len(columns)
    col_idx = {col: i for i, col in enumerate(columns)}
    adj = np.zeros((n, n), dtype=int)
    for src, tgt in edges:
        if src in col_idx and tgt in col_idx:
            adj[col_idx[src], col_idx[tgt]] = 1
    return adj

def generate_ci_queries(adj, columns, processed_columns):
    n = adj.shape[0]
    queries = []
    for i in range(n):
        for j in range(i+1, n):
            X, Y = columns[i], columns[j]
            X_cols = [col for col in processed_columns if col.startswith(f"{X}_") or col == X]
            Y_cols = [col for col in processed_columns if col.startswith(f"{Y}_") or col == Y]
            if X_cols and Y_cols:
                queries.append((X_cols[0], Y_cols[0], []))
    return queries

def run_ci_test(df, X, Y, S, num_cols, cat_cols):
    try:
        from causallearn.utils.cit import fisherz, kci
    except ImportError:
        fisherz = None
        kci = None
    if callable(fisherz) and all(col in num_cols for col in [X, Y] + S):
        return fisherz(df[X].values, df[Y].values, df[S].values if S else None)[1]
    elif callable(kci):
        return kci(df[X].values, df[Y].values, df[S].values if S else None)[1]
    else:
        from scipy.stats import pearsonr
        return pearsonr(df[X], df[Y])[1]

def get_base_col(col):
    return col.split('_')[0] if '_' in col else col

def compute_ci_auc(data, info, graph_path):
    categorical = info.get('categorical_features', [])
    num_cols, cat_cols = identify_variable_types(data, categorical)
    data_proc, _ = preprocess_for_ci(data, cat_cols)
    columns = data.columns.tolist()
    processed_columns = data_proc.columns.tolist()
    adj = load_ground_truth_graph(graph_path, columns)
    queries = generate_ci_queries(adj, columns, processed_columns)
    y_true = []
    y_score = []
    for X, Y, S in queries:
        X_base = get_base_col(X)
        Y_base = get_base_col(Y)
        i, j = columns.index(X_base), columns.index(Y_base)
        dsep = int(adj[i, j] == 0 and adj[j, i] == 0)
        y_true.append(dsep)
        pval = run_ci_test(data_proc, X, Y, S, num_cols, cat_cols)
        y_score.append(pval)
    auc = roc_auc_score(y_true, y_score)
    return auc, y_true, y_score
