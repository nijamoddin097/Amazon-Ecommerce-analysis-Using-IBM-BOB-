"""
Amazon E-Commerce 1M Dataset — Visualizations
===============================================
Run:  python analysis/visualize.py
Output: visualizations/*.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/amazon_ecommerce_cleaned.csv"
VIZ_DIR    = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["purchase_date"])

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

COLORS = sns.color_palette("tab10", 12)


def savefig(name):
    path = os.path.join(VIZ_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


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


print("[3/12] Top 10 Brands by Order Volume (Horizontal Bar)")
top_brands = df["brand"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top_brands.index[::-1], top_brands.values[::-1], color=COLORS[:10])
ax.set_title("Top 10 Brands by Order Volume", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Orders")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
savefig("03_top10_brands_orders.png")


print("[4/12] Return Rate by Category (Bar Chart)")
return_rate = df.groupby("category")["is_returned"].mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(return_rate.index, return_rate.values,
              color=["#e53935" if v > return_rate.mean() else "#43a047" for v in return_rate.values])
ax.axhline(return_rate.mean(), color="navy", linestyle="--", linewidth=1.2, label=f"Mean: {return_rate.mean():.1f}%")
ax.set_title("Return Rate by Category (%)", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Return Rate (%)")
ax.legend()
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
savefig("04_return_rate_by_category.png")


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


print("[8/12] Average Discount by Category (Bar Chart)")
avg_disc = df.groupby("category")["discount"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(avg_disc.index, avg_disc.values, color=COLORS[:len(avg_disc)])
ax.set_title("Average Discount % by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Average Discount (%)")
ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
savefig("08_avg_discount_by_category.png")


print("[9/12] Orders by Device Type (Bar Chart)")
device = df["device"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(device.index, device.values, color=COLORS[:len(device)])
ax.set_title("Orders by Device Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Device")
ax.set_ylabel("Number of Orders")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
savefig("09_orders_by_device.png")


print("[10/12] Correlation Heatmap")
num_cols = ["price", "discount", "final_price", "rating",
            "review_count", "stock", "seller_rating", "shipping_time_days"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
savefig("10_correlation_heatmap.png")


print("[11/12] Top 5 Cities by Revenue (Bar Chart)")
city_rev = df.groupby("location")["final_price"].sum().sort_values(ascending=False).head(5)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(city_rev.index, city_rev.values / 1e6, color=COLORS[:5])
ax.set_title("Top 5 Cities by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("City")
ax.set_ylabel("Revenue (Millions INR)")
ax.bar_label(bars, fmt="%.1fM", padding=3, fontsize=9)
savefig("11_top5_cities_revenue.png")


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


print("\n[DONE] All 12 visualizations saved to visualizations/")
