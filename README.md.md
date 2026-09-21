# Amazon E-Commerce Dataset Analysis

> **Dataset:** `amazon_ecommerce_1M.csv` — 1,000,000 rows × 20 columns  
> **Tech Stack:** Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Streamlit

---

## Project Structure

```
amazon_ecommerce_analysis/
│
├── data/
│   ├── amazon_ecommerce_1M.csv          # Raw dataset
│   └── amazon_ecommerce_cleaned.csv     # Cleaned dataset (output of cleaning step)
│
├── analysis/
│   ├── eda.py                           # EDA — Q&A analysis script
│   └── visualize.py                     # Visualization generation script
│
├── visualizations/                      # All PNG charts (auto-generated)
│   ├── 01_revenue_by_category.png
│   ├── 02_monthly_revenue_trend.png
│   ├── 03_top10_brands_orders.png
│   ├── 04_return_rate_by_category.png
│   ├── 05_rating_distribution.png
│   ├── 06_payment_method_pie.png
│   ├── 07_delivery_status_donut.png
│   ├── 08_avg_discount_by_category.png
│   ├── 09_orders_by_device.png
│   ├── 10_correlation_heatmap.png
│   ├── 11_top5_cities_revenue.png
│   └── 12_avg_order_value_category_device.png
│
├── reports/
│   └── eda_qa_report.txt                # Full EDA Q&A report (auto-generated)
│
├── output/                              # Reserved for additional exports
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Prerequisites

- Python 3.10 or higher
- pip

---

## Installation

```bash
# 1. Navigate to project folder
cd amazon_ecommerce_analysis

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Running the Analysis

### Step 1 — Run EDA (Q&A Text Report)
```bash
python analysis/eda.py
```
Generates: `reports/eda_qa_report.txt`  
Contains 30+ answered questions across 9 sections covering pricing, revenue,
returns, ratings, customer behaviour, inventory, and correlations.

---

### Step 2 — Generate All Visualizations
```bash
python analysis/visualize.py
```
Generates 12 charts in `visualizations/`:

| # | File | Description |
|---|------|-------------|
| 01 | `01_revenue_by_category.png` | Total revenue per category |
| 02 | `02_monthly_revenue_trend.png` | Monthly revenue over time |
| 03 | `03_top10_brands_orders.png` | Top 10 brands by order count |
| 04 | `04_return_rate_by_category.png` | Return rate % per category |
| 05 | `05_rating_distribution.png` | Product rating histogram |
| 06 | `06_payment_method_pie.png` | Payment method share |
| 07 | `07_delivery_status_donut.png` | Delivery status breakdown |
| 08 | `08_avg_discount_by_category.png` | Average discount by category |
| 09 | `09_orders_by_device.png` | Orders split by device type |
| 10 | `10_correlation_heatmap.png` | Feature correlation matrix |
| 11 | `11_top5_cities_revenue.png` | Top 5 cities by revenue |
| 12 | `12_avg_order_value_category_device.png` | AOV by category and device |

---

## Key Findings (Quick Summary)

| Insight | Finding |
|---------|---------|
| Dataset quality | Zero missing values, zero duplicates |
| Date range | Jan 2024 – Dec 2025 |
| Total revenue | ~Rs 9.8 Billion |
| Average discount | ~15% |
| Average rating | ~4.0 / 5 |
| Overall return rate | ~30% |
| Top category by revenue | Electronics |
| Most used payment method | UPI |
| Most used device | Mobile App |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.3 | Data loading & analysis |
| numpy | 2.1.3 | Numerical operations |
| matplotlib | 3.10.0 | Base plotting |
| seaborn | 0.13.2 | Statistical visualizations |
| scipy | 1.15.3 | Statistical computations |
| plotly | 5.24.1 | Interactive charts |
| streamlit | 1.45.1 | Web dashboard |

---

## License

For educational and analytical use only.
