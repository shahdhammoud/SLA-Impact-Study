# Visualization Features Documentation

## 📊 Overview

Your `ranking.py` now includes **5 comprehensive visualization methods** that automatically generate and save publication-quality plots!

---

## 🎨 Available Visualization Methods

### 1. **Reliability Heatmap** 
`plot_reliability_heatmap(save_path, figsize)`

**What it shows:** Matrix view of reliability scores for all algorithm-model combinations

**Features:**
- Color-coded heatmap (red = low reliability, green = high reliability)
- Annotated values showing exact percentages
- Easy to spot which algorithm is best for which model

**Example Output:**
```
        GMM    CTGAN   TabDDPM
PC      85.5   72.3    91.2
GES     45.6   38.9    52.1
FCI     67.2   71.8    69.5
```

**Usage:**
```python
comparator.plot_reliability_heatmap(
    save_path='outputs/rankings/asia_heatmap.png',
    figsize=(12, 8)
)
```

---

### 2. **Algorithm Comparison Bar Chart**
`plot_algorithm_comparison(save_path, figsize)`

**What it shows:** Average reliability of each algorithm across all models

**Features:**
- Color-coded bars (green = reliable, red = unreliable)
- Shows which algorithms are best for evaluation overall
- Value labels on top of each bar
- 50% threshold reference line

**Key Insight:** Tells you which algorithm to trust for synthetic data evaluation

**Usage:**
```python
comparator.plot_algorithm_comparison(
    save_path='outputs/rankings/algorithm_comparison.png',
    figsize=(14, 8)
)
```

---

### 3. **Model Rankings by Algorithm**
`plot_model_ranking_by_algorithm(save_path, figsize)`

**What it shows:** Side-by-side horizontal bar charts showing model rankings for each algorithm

**Features:**
- Separate panel for each algorithm
- Color-coded by score (green = good, red = poor)
- Easy comparison of how rankings change across algorithms
- Value labels with scores

**Key Insight:** Shows if model rankings are consistent across algorithms

**Usage:**
```python
comparator.plot_model_ranking_by_algorithm(
    save_path='outputs/rankings/model_rankings.png',
    figsize=(14, 10)
)
```

---

### 4. **CauTabBench vs Reliability Scatter Plot**
`plot_cautabench_vs_reliability(save_path, figsize)`

**What it shows:** Relationship between traditional CauTabBench scores and algorithm reliability

**Features:**
- Scatter plot with labeled points for each model
- Diagonal reference line (perfect correlation)
- Shows if CauTabBench agrees with structure-based evaluation

**Key Insight:** Validates if CauTabBench and structure reliability give similar rankings

**Usage:**
```python
comparator.plot_cautabench_vs_reliability(
    save_path='outputs/rankings/cautabench_vs_reliability.png',
    figsize=(12, 8)
)
```

---

### 5. **All Visualizations (One Command!)**
`plot_all_visualizations(output_dir, dataset_name)`

**What it does:** Generates ALL 4 plots above in one command!

**Features:**
- Automatically creates output directory
- Consistent file naming
- Progress messages
- Saves all plots at once

**Usage:**
```python
comparator.plot_all_visualizations(
    output_dir='outputs/rankings/visualizations',
    dataset_name='asia'
)
```

**Outputs:**
- `asia_reliability_heatmap.png`
- `asia_algorithm_comparison.png`
- `asia_model_rankings.png`
- `asia_cautabench_vs_reliability.png`

---

## 🚀 Quick Start

### **Option 1: Use the example script**

```bash
python example_ranking_visualization.py
```

### **Option 2: Use the compare rankings script**

```bash
python scripts/07_compare_rankings.py \
    --dataset asia \
    --algorithms pc,ges \
    --models gmm,ctgan,tabddpm \
    --visualize
```

### **Option 3: Python code**

```python
from src.evaluation.ranking import RankingComparator

# Load your data
comparator = RankingComparator()
comparator.load_ground_truth('pc', 'benchmarks_with_ground_truth/txt/asia.txt')
comparator.load_real_structure('pc', 'outputs/structures/asia_pc_real.json')
comparator.load_synthetic_structure('gmm', 'pc', 'outputs/structures/asia_pc_synthetic_gmm.json')

# Add CauTabBench scores
comparator.add_cautabench_score('gmm', 0.96)

# Compute scores
comparator.compute_all_reliability_scores()

# Generate ALL visualizations
comparator.plot_all_visualizations('outputs/rankings/viz', 'asia')

# Or generate individual plots
comparator.plot_reliability_heatmap('outputs/heatmap.png')
comparator.plot_algorithm_comparison('outputs/algo_compare.png')
comparator.plot_model_ranking_by_algorithm('outputs/model_ranks.png')
comparator.plot_cautabench_vs_reliability('outputs/scatter.png')
```

---

## 📁 Output Structure

After running visualizations, your directory will look like:

```
outputs/
  rankings/
    asia_ranking_comparison.json          # JSON results
    visualizations/
      asia_reliability_heatmap.png        # Heatmap
      asia_algorithm_comparison.png       # Algorithm bars
      asia_model_rankings.png             # Model rankings
      asia_cautabench_vs_reliability.png  # Scatter plot
```

---

## 🎯 Interpretation Guide

### **High Reliability Score (>80%)**
✅ Algorithm is RELIABLE for evaluating this model
✅ Trust the algorithm's assessment

### **Medium Reliability Score (50-80%)**
⚠️ Algorithm is MODERATELY reliable
⚠️ Use with caution, compare with other algorithms

### **Low Reliability Score (<50%)**
❌ Algorithm is UNRELIABLE for this model
❌ Do NOT trust this algorithm's assessment alone

---

## 🎨 Customization

All plot methods accept parameters:

```python
# Custom figure size
comparator.plot_reliability_heatmap(
    save_path='my_heatmap.png',
    figsize=(16, 10)  # Larger plot
)

# Just display without saving
comparator.plot_algorithm_comparison(
    save_path=None  # Will display instead of save
)
```

---

## 📊 Plot Specifications

All plots are saved with:
- **DPI:** 300 (publication quality)
- **Format:** PNG with transparency
- **Style:** Professional color schemes (RdYlGn, tab10)
- **Labels:** Clear, bold fonts
- **Grids:** Light gridlines for readability

---

## 💡 Use Cases

### **For Your Research Paper:**
1. Use **heatmap** in methods section (show algorithm-model comparison)
2. Use **algorithm comparison** to justify which algorithm you chose
3. Use **model rankings** to show your results
4. Use **scatter plot** to validate against CauTabBench

### **For Presentations:**
1. Start with **algorithm comparison** (which is most reliable?)
2. Show **model rankings** (which model wins?)
3. End with **heatmap** (comprehensive view)

### **For Reports:**
- Include all 4 plots in appendix
- Reference specific plots when discussing results

---

## ✅ Summary

**YES! Your code now has complete visualization capabilities:**

✅ 4 different plot types  
✅ Automatic generation and saving  
✅ Publication-quality output (300 DPI)  
✅ One-command option (`plot_all_visualizations()`)  
✅ Customizable sizes and paths  
✅ Professional color schemes  
✅ Clear labels and legends  

**All visualizations are automatically saved as PNG files! 🎉**

