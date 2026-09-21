"""
================================================================================
  Nijamoddin_E-commerce_Analysis.py
  Amazon E-Commerce 1M Dataset — Complete Analysis Pipeline
================================================================================
  Author  : Nijamoddin
  Dataset : amazon_ecommerce_1M.csv  (1,000,000 rows x 20 columns)

  This single file runs three stages in sequence:
    STAGE 1 — Data Cleaning       -> data/amazon_ecommerce_cleaned.csv
                                     reports/cleaning_report.txt
    STAGE 2 — EDA (Q&A Analysis)  -> reports/eda_qa_report.txt
    STAGE 3 — Visualizations      -> visualizations/*.png  (12 charts)

  Run:
    python Nijamoddin_E-commerce_Analysis.py
================================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("data",          exist_ok=True)
os.makedirs("reports",       exist_ok=True)
os.makedirs("visualizations",exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CSV      = "data/amazon_ecommerce_1M.csv"
CLEANED_CSV  = "data/amazon_ecommerce_cleaned.csv"
CLEAN_REPORT = "reports/cleaning_report.txt"
EDA_REPORT   = "reports/eda_qa_report.txt"
VIZ_DIR      = "visualizations"

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 1 — DATA CLEANING")
print("=" * 70)

df = pd.read_csv(RAW_CSV)

report_lines = []
report_lines.append("=" * 60)
report_lines.append("  AMAZON E-COMMERCE DATASET  -  CLEANING REPORT")
report_lines.append("=" * 60)
report_lines.append(f"\nOriginal shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

# -- 1. Missing values (before cleaning) --------------------------------------
report_lines.append("\n-- Missing Values (before cleaning) --------------------")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
missing_df  = missing_df[missing_df["missing_count"] > 0]

if missing_df.empty:
    report_lines.append("  [OK] No missing values found in any column.")
else:
    report_lines.append(missing_df.to_string())

report_lines.append(f"\n  Total cells with NaN : {df.isnull().sum().sum():,}")

# -- 2. Duplicate rows ---------------------------------------------------------
dup_count = df.duplicated().sum()
report_lines.append(f"\n-- Duplicate Rows ----------------------------------------")
report_lines.append(f"  Duplicate rows found : {dup_count:,}")

# -- 3. Data types -------------------------------------------------------------
report_lines.append("\n-- Column Data Types -------------------------------------")
report_lines.append(df.dtypes.to_string())

# -- 4. Fix purchase_date -> datetime -----------------------------------------
report_lines.append("\n-- Type Fixes --------------------------------------------")
before_null_dates = df["purchase_date"].isnull().sum()
df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="%d-%m-%y", errors="coerce")
after_null_dates  = df["purchase_date"].isnull().sum()
new_nulls = after_null_dates - before_null_dates
report_lines.append(f"  purchase_date -> datetime64  (unparseable rows set to NaT: {new_nulls})")

# -- 5. is_returned -> bool ----------------------------------------------------
df["is_returned"] = (
    df["is_returned"].astype(str).str.strip().str.upper()
    .map({"TRUE": True, "FALSE": False})
)
report_lines.append("  is_returned -> bool")

# -- 6. Numeric sanity checks --------------------------------------------------
report_lines.append("\n-- Outlier / Sanity Checks -------------------------------")
neg_price    = (df["price"] < 0).sum()
neg_final    = (df["final_price"] < 0).sum()
neg_discount = (df["discount"] < 0).sum()
rating_oob   = ((df["rating"] < 0) | (df["rating"] > 5)).sum()
neg_stock    = (df["stock"] < 0).sum()
neg_ship     = (df["shipping_time_days"] < 0).sum()

report_lines.append(f"  price < 0              : {neg_price}")
report_lines.append(f"  final_price < 0        : {neg_final}")
report_lines.append(f"  discount < 0           : {neg_discount}")
report_lines.append(f"  rating out of [0,5]    : {rating_oob}")
report_lines.append(f"  stock < 0              : {neg_stock}")
report_lines.append(f"  shipping_time_days < 0 : {neg_ship}")

bad_rows = (df["price"] < 0) | (df["final_price"] < 0)
if bad_rows.sum() > 0:
    df = df[~bad_rows]
    report_lines.append(f"\n  [REMOVED] {bad_rows.sum()} rows with negative price/final_price.")

# -- 7. Strip whitespace from string columns ----------------------------------
str_cols = df.select_dtypes(include="object").columns
for col in str_cols:
    df[col] = df[col].str.strip()
report_lines.append(f"\n-- Whitespace stripped from {len(str_cols)} string columns.")

# -- 8. Drop duplicates -------------------------------------------------------
df.drop_duplicates(inplace=True)
report_lines.append(f"-- Duplicates removed : {dup_count}")

# -- 9. Fill remaining NaNs ---------------------------------------------------
num_cols = df.select_dtypes(include=[np.number]).columns
filled   = 0
for col in num_cols:
    n = df[col].isnull().sum()
    if n > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        report_lines.append(f"  Filled {n} NaN in '{col}' with median ({median_val:.2f})")
        filled += n
if filled == 0:
    report_lines.append("-- No numeric NaNs to fill.")

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    n = df[col].isnull().sum()
    if n > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        report_lines.append(f"  Filled {n} NaN in '{col}' with mode ('{mode_val}')")

# -- 10. Final state + save ---------------------------------------------------
report_lines.append(f"\n-- Final State -------------------------------------------")
report_lines.append(f"  Cleaned shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")
report_lines.append(f"  Remaining NaN (total)  : {df.isnull().sum().sum()}")
report_lines.append(f"  Remaining duplicates   : {df.duplicated().sum()}")

df.to_csv(CLEANED_CSV, index=False)
report_lines.append(f"\n  [SAVED] Cleaned file -> {CLEANED_CSV}")

clean_report_text = "\n".join(report_lines)
with open(CLEAN_REPORT, "w", encoding="utf-8") as f:
    f.write(clean_report_text)

print(clean_report_text)
print(f"\n[STAGE 1 DONE] Cleaned CSV -> {CLEANED_CSV}")
print(f"               Cleaning report -> {CLEAN_REPORT}")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — EDA (Q&A ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 2 — EDA Q&A ANALYSIS")
print("=" * 70)

df = pd.read_csv(CLEANED_CSV, parse_dates=["purchase_date"])

eda_lines = []

def h(title):
    eda_lines.append("\n" + "=" * 70)
    eda_lines.append(f"  {title}")
    eda_lines.append("=" * 70)

def q(question, answer):
    eda_lines.append(f"\nQ: {question}")
    eda_lines.append(f"A: {answer}")

# -- Section 1: Dataset Overview ----------------------------------------------
h("SECTION 1 — DATASET OVERVIEW")
q("How many records and columns does the dataset have?",
  f"{df.shape[0]:,} rows x {df.shape[1]} columns.")
q("What are the column names and data types?",
  df.dtypes.to_string())
q("What is the date range of purchases?",
  f"{df['purchase_date'].min().date()} to {df['purchase_date'].max().date()}")
q("How many unique users and products are there?",
  f"Unique users: {df['user_id'].nunique():,}  |  Unique products: {df['product_id'].nunique():,}")
q("How many unique categories and subcategories are there?",
  f"Categories: {df['category'].nunique()}  |  Subcategories: {df['subcategory'].nunique()}")
q("How many unique sellers are there?",
  f"{df['seller_id'].nunique():,} unique sellers")

# -- Section 2: Pricing & Discount --------------------------------------------
h("SECTION 2 — PRICING & DISCOUNT ANALYSIS")
q("What is the average, min, and max product price?",
  f"Mean: {df['price'].mean():,.2f}  |  Min: {df['price'].min():,.2f}  |  Max: {df['price'].max():,.2f}")
q("What is the average discount percentage across all products?",
  f"{df['discount'].mean():.2f}%")
q("Which category offers the highest average discount?",
  df.groupby("category")["discount"].mean().sort_values(ascending=False)
    .rename_axis("category").reset_index(name="avg_discount").to_string(index=False))
q("What is the average final price per category?",
  df.groupby("category")["final_price"].mean().sort_values(ascending=False).round(2).to_string())
q("Which top 5 brands have the highest average price?",
  df.groupby("brand")["price"].mean().sort_values(ascending=False).head(5).round(2).to_string())
q("What percentage of orders had a discount greater than 20%?",
  f"{(df['discount'] > 20).mean() * 100:.2f}%")

# -- Section 3: Sales & Revenue -----------------------------------------------
h("SECTION 3 — SALES & REVENUE ANALYSIS")
q("What is the total revenue (sum of final_price)?",
  f"Rs {df['final_price'].sum():,.2f}")
q("Which category generates the most revenue?",
  df.groupby("category")["final_price"].sum().sort_values(ascending=False).round(2).to_string())
q("Which are the top 5 selling subcategories by order count?",
  df["subcategory"].value_counts().head(5).to_string())
q("What is monthly revenue trend?",
  df.groupby(df["purchase_date"].dt.to_period("M"))["final_price"].sum().round(2).to_string())
q("Which location (city) has the highest total spending?",
  df.groupby("location")["final_price"].sum().sort_values(ascending=False).round(2).to_string())
q("Which payment method is most popular?",
  df["payment_method"].value_counts().to_string())

# -- Section 4: Ratings & Reviews ---------------------------------------------
h("SECTION 4 — RATINGS & REVIEWS")
q("What is the overall average product rating?",
  f"{df['rating'].mean():.2f} out of 5")
q("What is the distribution of ratings?",
  df["rating"].value_counts().sort_index().to_string())
q("Which category has the highest average rating?",
  df.groupby("category")["rating"].mean().sort_values(ascending=False).round(2).to_string())
q("Which brand has the highest average rating (with >= 1000 reviews)?",
  df.groupby("brand").filter(lambda x: x["review_count"].sum() >= 1000)
    .groupby("brand")["rating"].mean().sort_values(ascending=False).head(5).round(2).to_string())
q("What is the average review count per category?",
  df.groupby("category")["review_count"].mean().sort_values(ascending=False).round(1).to_string())

# -- Section 5: Returns & Delivery --------------------------------------------
h("SECTION 5 — RETURNS & DELIVERY ANALYSIS")
q("What is the overall product return rate?",
  f"{df['is_returned'].mean() * 100:.2f}%")
q("Which category has the highest return rate?",
  df.groupby("category")["is_returned"].mean().sort_values(ascending=False)
    .mul(100).round(2).rename("return_rate_%").to_string())
q("What is the distribution of delivery statuses?",
  df["delivery_status"].value_counts().to_string())
q("What is the average shipping time (days) per delivery status?",
  df.groupby("delivery_status")["shipping_time_days"].mean().round(2).to_string())
q("Which location has the highest return rate?",
  df.groupby("location")["is_returned"].mean().sort_values(ascending=False)
    .mul(100).round(2).rename("return_rate_%").to_string())
q("Do returned items have lower ratings on average?",
  f"Returned:     {df[df['is_returned']==True]['rating'].mean():.2f}\n"
  f"  Not returned: {df[df['is_returned']==False]['rating'].mean():.2f}")

# -- Section 6: Customer Behaviour --------------------------------------------
h("SECTION 6 — CUSTOMER BEHAVIOUR")
q("Which device is most used for purchases?",
  df["device"].value_counts().to_string())
q("Which device has the highest average order value?",
  df.groupby("device")["final_price"].mean().sort_values(ascending=False).round(2).to_string())
q("Which payment method has the highest average order value?",
  df.groupby("payment_method")["final_price"].mean().sort_values(ascending=False).round(2).to_string())
q("What is the purchase volume by day of week?",
  df.groupby(df["purchase_date"].dt.day_name())["user_id"].count()
    .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).to_string())
q("What is the average seller rating per category?",
  df.groupby("category")["seller_rating"].mean().sort_values(ascending=False).round(2).to_string())

# -- Section 7: Stock & Inventory ---------------------------------------------
h("SECTION 7 — STOCK & INVENTORY")
q("What is the average stock level per category?",
  df.groupby("category")["stock"].mean().sort_values(ascending=False).round(1).to_string())
q("How many products are low-stock (stock < 10)?",
  f"{(df['stock'] < 10).sum():,} records ({(df['stock'] < 10).mean()*100:.2f}%)")
q("Which subcategory has the lowest average stock?",
  df.groupby("subcategory")["stock"].mean().sort_values().head(5).round(1).to_string())

# -- Section 8: Correlation Summary -------------------------------------------
h("SECTION 8 — CORRELATION SUMMARY")
num_cols_corr = ["price", "discount", "final_price", "rating",
                 "review_count", "stock", "seller_rating", "shipping_time_days"]
corr = df[num_cols_corr].corr().round(3)
q("Correlation matrix of numeric features:", corr.to_string())
q("Is there a correlation between discount and rating?",
  f"Pearson r = {df['discount'].corr(df['rating']):.4f}  "
  f"(weak {'positive' if df['discount'].corr(df['rating']) > 0 else 'negative'} relationship)")
q("Is there a correlation between price and review count?",
  f"Pearson r = {df['price'].corr(df['review_count']):.4f}")

# -- Section 9: Top Performers ------------------------------------------------
h("SECTION 9 — TOP PERFORMERS")
q("Which are the top 5 categories by order volume?",
  df["category"].value_counts().head(5).to_string())
q("Which are the top 5 brands by order volume?",
  df["brand"].value_counts().head(5).to_string())
q("Which are the top 5 cities by order count?",
  df["location"].value_counts().head(5).to_string())
q("Which are the top 5 sellers by revenue?",
  df.groupby("seller_id")["final_price"].sum().sort_values(ascending=False)
    .head(5).round(2).to_string())

# -- Save EDA report ----------------------------------------------------------
eda_text = "\n".join(eda_lines)
with open(EDA_REPORT, "w", encoding="utf-8") as f:
    f.write(eda_text)

print(eda_text)
print(f"\n[STAGE 2 DONE] EDA report -> {EDA_REPORT}")
print(f"               Total questions answered: {eda_text.count(chr(10)+'Q:')}")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — VISUALIZATIONS  (12 charts)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 3 — VISUALIZATIONS")
print("=" * 70)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
COLORS = sns.color_palette("tab10", 12)

def savefig(name):
    path = os.path.join(VIZ_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# [1] Revenue by Category
print("[1/12] Revenue by Category (Bar Chart)")
rev_cat = df.groupby("category")["final_price"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(rev_cat.index, rev_cat.values / 1e6, color=COLORS[:len(rev_cat)])
ax.set_title("Total Revenue by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Revenue (Millions INR)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
ax.bar_label(bars, fmt="%.1fM", padding=3, fontsize=8)
savefig("01_revenue_by_category.png")

# [2] Monthly Revenue Trend
print("[2/12] Monthly Revenue Trend (Line Chart)")
monthly = df.groupby(df["purchase_date"].dt.to_period("M"))["final_price"].sum().reset_index()
monthly["purchase_date"] = monthly["purchase_date"].astype(str)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["purchase_date"], monthly["final_price"] / 1e6,
        marker="o", linewidth=2, color="#2196F3", markersize=4)
ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (Millions INR)")
ax.tick_params(axis="x", rotation=45)
ax.fill_between(range(len(monthly)), monthly["final_price"] / 1e6, alpha=0.1, color="#2196F3")
savefig("02_monthly_revenue_trend.png")

# [3] Top 10 Brands by Order Volume
print("[3/12] Top 10 Brands by Order Volume (Horizontal Bar)")
top_brands = df["brand"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top_brands.index[::-1], top_brands.values[::-1], color=COLORS[:10])
ax.set_title("Top 10 Brands by Order Volume", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Orders")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
savefig("03_top10_brands_orders.png")

# [4] Return Rate by Category
print("[4/12] Return Rate by Category (Bar Chart)")
return_rate = df.groupby("category")["is_returned"].mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(return_rate.index, return_rate.values,
              color=["#e53935" if v > return_rate.mean() else "#43a047" for v in return_rate.values])
ax.axhline(return_rate.mean(), color="navy", linestyle="--", linewidth=1.2,
           label=f"Mean: {return_rate.mean():.1f}%")
ax.set_title("Return Rate by Category (%)", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Return Rate (%)")
ax.legend()
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
savefig("04_return_rate_by_category.png")

# [5] Rating Distribution
print("[5/12] Rating Distribution (Histogram)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["rating"], bins=30, color="#7E57C2", edgecolor="white", linewidth=0.5)
ax.axvline(df["rating"].mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Mean: {df['rating'].mean():.2f}")
ax.set_title("Product Rating Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
ax.legend()
savefig("05_rating_distribution.png")

# [6] Payment Method Distribution
print("[6/12] Payment Method Distribution (Pie Chart)")
pm = df["payment_method"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    pm.values, labels=pm.index, autopct="%1.1f%%",
    colors=COLORS[:len(pm)], startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Payment Method Distribution", fontsize=14, fontweight="bold")
savefig("06_payment_method_pie.png")

# [7] Delivery Status Donut
print("[7/12] Delivery Status Distribution (Donut Chart)")
ds = df["delivery_status"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    ds.values, labels=ds.index, autopct="%1.1f%%",
    colors=COLORS[:len(ds)], startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.65}
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Delivery Status Distribution", fontsize=14, fontweight="bold")
savefig("07_delivery_status_donut.png")

# [8] Average Discount by Category
print("[8/12] Average Discount by Category (Bar Chart)")
avg_disc = df.groupby("category")["discount"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(avg_disc.index, avg_disc.values, color=COLORS[:len(avg_disc)])
ax.set_title("Average Discount % by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Average Discount (%)")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
savefig("08_avg_discount_by_category.png")

# [9] Orders by Device Type
print("[9/12] Orders by Device Type (Bar Chart)")
device = df["device"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(device.index, device.values, color=COLORS[:len(device)])
ax.set_title("Orders by Device Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Device")
ax.set_ylabel("Number of Orders")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
savefig("09_orders_by_device.png")

# [10] Correlation Heatmap
print("[10/12] Correlation Heatmap")
num_cols_viz = ["price", "discount", "final_price", "rating",
                "review_count", "stock", "seller_rating", "shipping_time_days"]
corr_viz = df[num_cols_viz].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_viz, dtype=bool))
sns.heatmap(corr_viz, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
savefig("10_correlation_heatmap.png")

# [11] Top 5 Cities by Revenue
print("[11/12] Top 5 Cities by Revenue (Bar Chart)")
city_rev = df.groupby("location")["final_price"].sum().sort_values(ascending=False).head(5)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(city_rev.index, city_rev.values / 1e6, color=COLORS[:5])
ax.set_title("Top 5 Cities by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("City")
ax.set_ylabel("Revenue (Millions INR)")
ax.bar_label(bars, fmt="%.1fM", padding=3, fontsize=9)
savefig("11_top5_cities_revenue.png")

# [12] Avg Order Value by Category & Device
print("[12/12] Average Order Value by Category & Device (Grouped Bar)")
pivot = df.groupby(["category", "device"])["final_price"].mean().unstack()
fig, ax = plt.subplots(figsize=(13, 6))
pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", linewidth=0.5)
ax.set_title("Average Order Value by Category and Device", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Avg Final Price (INR)")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Device", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
savefig("12_avg_order_value_category_device.png")

print(f"\n[STAGE 3 DONE] All 12 charts saved to {VIZ_DIR}/")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ALL STAGES COMPLETE")
print("=" * 70)
print(f"  Cleaned CSV     -> {CLEANED_CSV}")
print(f"  Cleaning report -> {CLEAN_REPORT}")
print(f"  EDA Q&A report  -> {EDA_REPORT}")
print(f"  Visualizations  -> {VIZ_DIR}/ (12 charts)")
print("=" * 70)
