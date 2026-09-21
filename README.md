amazon_ecommerce_analysis/
├── data/
│   ├── amazon_ecommerce_1M.csv                    # Raw dataset (1M rows)
│   └── amazon_ecommerce_cleaned.csv               # Cleaned dataset
├── analysis/
│   ├── clean_dataset.py                           # Data cleaning script
│   ├── eda.py                                     # EDA — 44 Q&A questions
│   └── visualize.py                               # 12 chart generation
├── Nijamoddin_E-commerce_Analysis.py             # Merged single-file pipeline
├── visualizations/                                # 12 PNG charts (auto-generated)
├── reports/
│   ├── cleaning_report.txt
│   └── eda_qa_report.txt
├── requirements.txt
└── README.md


**Key Business Insights**
Electronics Dominance: 20% of order volume but 66% of total revenue — driven by high unit price (avg Rs 32,889).
High Delay Rate: ~30% of all orders are delayed — a critical logistics issue requiring attention.
Discount Trap: Clothing has the highest discounts (40%) yet the lowest revenue — heavy discounting doesn't drive revenue growth.
Payment Parity: All four payment methods share ~25% each — build and maintain support for all equally.
Balanced Geography: Revenue is nearly uniform across all 5 cities (~Rs 2B each) — no single dominant market.
Return Rate: 11.6% overall. Beauty and Clothing most returned — size/fit guides and better product descriptions recommended.
