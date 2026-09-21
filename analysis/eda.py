"""
Amazon E-Commerce 1M Dataset — Exploratory Data Analysis
=========================================================
Run:  python analysis/eda.py
Output: reports/eda_qa_report.txt
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/amazon_ecommerce_cleaned.csv"
REPORT_PATH = "reports/eda_qa_report.txt"
os.makedirs("reports", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["purchase_date"])

lines = []

def h(title):
    lines.append("\n" + "=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)

def q(question, answer):
    lines.append(f"\nQ: {question}")
    lines.append(f"A: {answer}")

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 1 — DATASET OVERVIEW")
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 2 — PRICING & DISCOUNT ANALYSIS")
# ══════════════════════════════════════════════════════════════════════════════

q("What is the average, min, and max product price?",
  f"Mean: {df['price'].mean():,.2f}  |  Min: {df['price'].min():,.2f}  |  Max: {df['price'].max():,.2f}")

q("What is the average discount percentage across all products?",
  f"{df['discount'].mean():.2f}%")

q("Which category offers the highest average discount?",
  df.groupby("category")["discount"].mean().sort_values(ascending=False)
    .rename_axis("category").reset_index(name="avg_discount")
    .to_string(index=False))

q("What is the average final price per category?",
  df.groupby("category")["final_price"].mean().sort_values(ascending=False)
    .round(2).to_string())

q("Which top 5 brands have the highest average price?",
  df.groupby("brand")["price"].mean().sort_values(ascending=False)
    .head(5).round(2).to_string())

q("What percentage of orders had a discount greater than 20%?",
  f"{(df['discount'] > 20).mean() * 100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 3 — SALES & REVENUE ANALYSIS")
# ══════════════════════════════════════════════════════════════════════════════

q("What is the total revenue (sum of final_price)?",
  f"Rs {df['final_price'].sum():,.2f}")

q("Which category generates the most revenue?",
  df.groupby("category")["final_price"].sum().sort_values(ascending=False)
    .round(2).to_string())

q("Which are the top 5 selling subcategories by order count?",
  df["subcategory"].value_counts().head(5).to_string())

q("What is monthly revenue trend?",
  df.groupby(df["purchase_date"].dt.to_period("M"))["final_price"]
    .sum().round(2).to_string())

q("Which location (city) has the highest total spending?",
  df.groupby("location")["final_price"].sum().sort_values(ascending=False)
    .round(2).to_string())

q("Which payment method is most popular?",
  df["payment_method"].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 4 — RATINGS & REVIEWS")
# ══════════════════════════════════════════════════════════════════════════════

q("What is the overall average product rating?",
  f"{df['rating'].mean():.2f} out of 5")

q("What is the distribution of ratings?",
  df["rating"].value_counts().sort_index().to_string())

q("Which category has the highest average rating?",
  df.groupby("category")["rating"].mean().sort_values(ascending=False)
    .round(2).to_string())

q("Which brand has the highest average rating (with >= 1000 reviews)?",
  df.groupby("brand").filter(lambda x: x["review_count"].sum() >= 1000)
    .groupby("brand")["rating"].mean().sort_values(ascending=False)
    .head(5).round(2).to_string())

q("What is the average review count per category?",
  df.groupby("category")["review_count"].mean().sort_values(ascending=False)
    .round(1).to_string())

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 5 — RETURNS & DELIVERY ANALYSIS")
# ══════════════════════════════════════════════════════════════════════════════

q("What is the overall product return rate?",
  f"{df['is_returned'].mean() * 100:.2f}%")

q("Which category has the highest return rate?",
  df.groupby("category")["is_returned"].mean().sort_values(ascending=False)
    .mul(100).round(2).rename("return_rate_%").to_string())

q("What is the distribution of delivery statuses?",
  df["delivery_status"].value_counts().to_string())

q("What is the average shipping time (days) per delivery status?",
  df.groupby("delivery_status")["shipping_time_days"].mean()
    .round(2).to_string())

q("Which location has the highest return rate?",
  df.groupby("location")["is_returned"].mean().sort_values(ascending=False)
    .mul(100).round(2).rename("return_rate_%").to_string())

q("Do returned items have lower ratings on average?",
  f"Returned:     {df[df['is_returned']==True]['rating'].mean():.2f}\n"
  f"  Not returned: {df[df['is_returned']==False]['rating'].mean():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 6 — CUSTOMER BEHAVIOUR")
# ══════════════════════════════════════════════════════════════════════════════

q("Which device is most used for purchases?",
  df["device"].value_counts().to_string())

q("Which device has the highest average order value?",
  df.groupby("device")["final_price"].mean().sort_values(ascending=False)
    .round(2).to_string())

q("Which payment method has the highest average order value?",
  df.groupby("payment_method")["final_price"].mean()
    .sort_values(ascending=False).round(2).to_string())

q("What is the purchase volume by day of week?",
  df.groupby(df["purchase_date"].dt.day_name())["user_id"].count()
    .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    .to_string())

q("What is the average seller rating per category?",
  df.groupby("category")["seller_rating"].mean().sort_values(ascending=False)
    .round(2).to_string())

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 7 — STOCK & INVENTORY")
# ══════════════════════════════════════════════════════════════════════════════

q("What is the average stock level per category?",
  df.groupby("category")["stock"].mean().sort_values(ascending=False)
    .round(1).to_string())

q("How many products are low-stock (stock < 10)?",
  f"{(df['stock'] < 10).sum():,} records ({(df['stock'] < 10).mean()*100:.2f}%)")

q("Which subcategory has the lowest average stock?",
  df.groupby("subcategory")["stock"].mean().sort_values().head(5)
    .round(1).to_string())

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 8 — CORRELATION SUMMARY")
# ══════════════════════════════════════════════════════════════════════════════

num_cols = ["price", "discount", "final_price", "rating",
            "review_count", "stock", "seller_rating", "shipping_time_days"]

corr = df[num_cols].corr().round(3)
q("Correlation matrix of numeric features:", corr.to_string())

q("Is there a correlation between discount and rating?",
  f"Pearson r = {df['discount'].corr(df['rating']):.4f}  "
  f"(weak {'positive' if df['discount'].corr(df['rating']) > 0 else 'negative'} relationship)")

q("Is there a correlation between price and review count?",
  f"Pearson r = {df['price'].corr(df['review_count']):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
h("SECTION 9 — TOP PERFORMERS")
# ══════════════════════════════════════════════════════════════════════════════

q("Which are the top 5 categories by order volume?",
  df["category"].value_counts().head(5).to_string())

q("Which are the top 5 brands by order volume?",
  df["brand"].value_counts().head(5).to_string())

q("Which are the top 5 cities by order count?",
  df["location"].value_counts().head(5).to_string())

q("Which are the top 5 sellers by revenue?",
  df.groupby("seller_id")["final_price"].sum().sort_values(ascending=False)
    .head(5).round(2).to_string())

# ── Write report ──────────────────────────────────────────────────────────────
report_text = "\n".join(lines)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"[OK] EDA Q&A report saved -> {REPORT_PATH}")
print(f"     Total questions answered: {report_text.count(chr(10)+'Q:')}")
