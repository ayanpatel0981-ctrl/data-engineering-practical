"""
========================================================================================
PRACTICAL 10: END-TO-END FINTECH TRANSACTION MONITORING & DATA WAREHOUSE (MINI PROJECT)
========================================================================================
Course: Data Engineering
Author: Ayan Patel (Solo Project)

Components:
  1. Data Ingestion (Heterogeneous Sources: CSV, JSON, Authentication Logs)
  2. Data Quality & Cleaning (Silver Layer Quarantine Architecture)
  3. Transformation & Risk Scoring (Transaction Fees & Account Risk Tiers)
  4. Data Warehouse Design (Star Schema in SQLite with Analytical Data Marts)
  5. Executive Reporting & Visual Analytics Dashboard
========================================================================================
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_SILVER = os.path.join(BASE_DIR, "data", "silver")
DATA_GOLD = os.path.join(BASE_DIR, "data", "gold")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for d in [DATA_RAW, DATA_SILVER, DATA_GOLD, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)


def run_pipeline():
    print("=" * 80)
    print("FINTECH TRANSACTION MONITORING & DATA WAREHOUSE PIPELINE")
    print("Author: Ayan Patel")
    print("=" * 80)

    # 1. INGESTION
    print("\n[STAGE 1] Ingesting Raw Multi-Source Data...")
    accounts_raw = pd.read_csv(os.path.join(DATA_RAW, "accounts.csv"))
    with open(os.path.join(DATA_RAW, "merchants.json"), "r") as f:
        merchants_raw = pd.DataFrame(json.load(f))
    tx_raw = pd.read_csv(os.path.join(DATA_RAW, "transactions.csv"))
    with open(os.path.join(DATA_RAW, "auth_logs.json"), "r") as f:
        auth_logs_raw = pd.DataFrame(json.load(f))

    print(f"  ✓ Ingested accounts.csv:      {len(accounts_raw)} records")
    print(f"  ✓ Ingested merchants.json:    {len(merchants_raw)} records")
    print(f"  ✓ Ingested transactions.csv:  {len(tx_raw)} records")
    print(f"  ✓ Ingested auth_logs.json:    {len(auth_logs_raw)} records")

    # 2. DATA QUALITY & QUARANTINE
    print("\n[STAGE 2] Enforcing Data Quality & Quarantine Logic...")
    quarantine = []

    # Clean Accounts
    acc_dedup = accounts_raw.drop_duplicates(subset=["account_id"], keep="first").copy()
    clean_acc_list = []
    for _, r in acc_dedup.iterrows():
        if r["balance"] < 0 or r["kyc_status"] != "Verified":
            quarantine.append({
                "record_type": "Account",
                "identifier": str(r["account_id"]),
                "reason": "Negative balance or unverified KYC status",
                "raw_data": str(r.to_dict())
            })
        else:
            row_dict = r.to_dict()
            row_dict["city"] = str(r["city"]).strip().title()
            clean_acc_list.append(row_dict)
    clean_accounts = pd.DataFrame(clean_acc_list)

    # Clean Merchants
    valid_merchants = []
    for _, r in merchants_raw.iterrows():
        if pd.isna(r["category"]) or r["commission_rate"] <= 0:
            quarantine.append({
                "record_type": "Merchant",
                "identifier": str(r["merchant_id"]),
                "reason": "Missing category or invalid negative commission rate",
                "raw_data": str(r.to_dict())
            })
        else:
            valid_merchants.append(r)
    clean_merchants = pd.DataFrame(valid_merchants)

    # Clean Transactions
    valid_acc_ids = set(clean_accounts["account_id"])
    valid_merch_ids = set(clean_merchants["merchant_id"])
    tx_dedup = tx_raw.drop_duplicates(subset=["transaction_id"], keep="first").copy()
    clean_tx_list = []

    for _, r in tx_dedup.iterrows():
        tid = r["transaction_id"]
        aid = r["account_id"]
        mid = r["merchant_id"]
        amt = r["amount"]
        dt_str = str(r["trans_date"])

        try:
            dt_parsed = pd.to_datetime(dt_str)
        except Exception:
            dt_parsed = pd.NaT

        if pd.isna(dt_parsed):
            quarantine.append({"record_type": "Transaction", "identifier": tid, "reason": f"Corrupted date: {dt_str}", "raw_data": str(r.to_dict())})
            continue
        if amt <= 0:
            quarantine.append({"record_type": "Transaction", "identifier": tid, "reason": f"Non-positive amount: {amt}", "raw_data": str(r.to_dict())})
            continue
        if aid not in valid_acc_ids:
            quarantine.append({"record_type": "Transaction", "identifier": tid, "reason": f"FK violation: Account {aid} not found", "raw_data": str(r.to_dict())})
            continue
        if mid not in valid_merch_ids:
            quarantine.append({"record_type": "Transaction", "identifier": tid, "reason": f"FK violation: Merchant {mid} not found", "raw_data": str(r.to_dict())})
            continue

        row_dict = r.to_dict()
        row_dict["trans_date"] = dt_parsed.strftime("%Y-%m-%d")
        clean_tx_list.append(row_dict)

    clean_tx = pd.DataFrame(clean_tx_list)
    df_quarantine = pd.DataFrame(quarantine)

    clean_accounts.to_csv(os.path.join(DATA_SILVER, "clean_accounts.csv"), index=False)
    clean_merchants.to_csv(os.path.join(DATA_SILVER, "clean_merchants.csv"), index=False)
    clean_tx.to_csv(os.path.join(DATA_SILVER, "clean_transactions.csv"), index=False)
    df_quarantine.to_csv(os.path.join(DATA_SILVER, "quarantine_audit_log.csv"), index=False)
    print(f"  ✓ Silver clean tables saved. Quarantined {len(df_quarantine)} defective records.")

    # 3. TRANSFORMATION & RISK SCORING
    print("\n[STAGE 3] Performing Financial Calculations & Risk Profiling...")
    enriched_tx = clean_tx.merge(clean_merchants[["merchant_id", "category", "risk_level", "commission_rate"]], on="merchant_id", how="inner")
    enriched_tx["fee_amount"] = enriched_tx["amount"] * enriched_tx["commission_rate"]
    enriched_tx["net_amount"] = enriched_tx["amount"] - enriched_tx["fee_amount"]
    enriched_tx["is_high_value"] = (enriched_tx["amount"] >= 25000).astype(int)

    acc_agg = enriched_tx.groupby("account_id").agg(
        total_spend=("amount", "sum"),
        tx_count=("transaction_id", "count"),
        high_value_count=("is_high_value", "sum")
    ).reset_index()

    def assign_risk(row):
        if row["total_spend"] >= 400000:
            return "High Net Worth (Tier 1)"
        elif row["high_value_count"] >= 3:
            return "High Volume Transactor (Tier 2)"
        elif row["tx_count"] >= 8:
            return "Standard Active (Tier 3)"
        return "Low Activity (Tier 4)"

    acc_agg["risk_tier"] = acc_agg.apply(assign_risk, axis=1)
    clean_accounts = clean_accounts.merge(acc_agg[["account_id", "risk_tier"]], on="account_id", how="left")
    clean_accounts["risk_tier"] = clean_accounts["risk_tier"].fillna("New Account")

    # 4. STAR SCHEMA DWH
    print("\n[STAGE 4] Loading Star Schema Data Warehouse (fintech_dwh.db)...")
    dwh_file = os.path.join(DATA_GOLD, "fintech_dwh.db")
    if os.path.exists(dwh_file):
        os.remove(dwh_file)
    conn = sqlite3.connect(dwh_file)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE dim_date (
        date_key INTEGER PRIMARY KEY,
        full_date TEXT NOT NULL UNIQUE,
        day_number INTEGER,
        month_number INTEGER,
        month_name TEXT,
        quarter INTEGER,
        year_number INTEGER,
        day_name TEXT,
        is_weekend INTEGER
    );
    CREATE TABLE dim_account (
        account_key INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT NOT NULL UNIQUE,
        holder_name TEXT NOT NULL,
        account_type TEXT,
        city TEXT,
        kyc_status TEXT,
        risk_tier TEXT
    );
    CREATE TABLE dim_merchant (
        merchant_key INTEGER PRIMARY KEY AUTOINCREMENT,
        merchant_id TEXT NOT NULL UNIQUE,
        merchant_name TEXT NOT NULL,
        category TEXT,
        risk_level TEXT,
        commission_rate REAL
    );
    CREATE TABLE dim_channel (
        channel_key INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_name TEXT NOT NULL,
        status TEXT NOT NULL,
        UNIQUE(channel_name, status)
    );
    CREATE TABLE fact_transactions (
        trans_key INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL UNIQUE,
        date_key INTEGER NOT NULL,
        account_key INTEGER NOT NULL,
        merchant_key INTEGER NOT NULL,
        channel_key INTEGER NOT NULL,
        amount REAL,
        fee_amount REAL,
        net_amount REAL,
        is_high_value INTEGER,
        FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
        FOREIGN KEY (account_key) REFERENCES dim_account(account_key),
        FOREIGN KEY (merchant_key) REFERENCES dim_merchant(merchant_key),
        FOREIGN KEY (channel_key) REFERENCES dim_channel(channel_key)
    );
    """)

    date_range = pd.date_range("2024-01-01", "2024-10-31", freq="D")
    dim_dates = [(int(d.strftime("%Y%m%d")), d.strftime("%Y-%m-%d"), d.day, d.month, d.strftime("%B"), (d.month-1)//3+1, d.year, d.strftime("%A"), 1 if d.weekday()>=5 else 0) for d in date_range]
    cur.executemany("INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", dim_dates)

    for _, r in clean_accounts.iterrows():
        cur.execute("INSERT INTO dim_account (account_id, holder_name, account_type, city, kyc_status, risk_tier) VALUES (?, ?, ?, ?, ?, ?)", (r["account_id"], r["holder_name"], r["account_type"], r["city"], r["kyc_status"], r["risk_tier"]))
    for _, r in clean_merchants.iterrows():
        cur.execute("INSERT INTO dim_merchant (merchant_id, merchant_name, category, risk_level, commission_rate) VALUES (?, ?, ?, ?, ?)", (r["merchant_id"], r["merchant_name"], r["category"], r["risk_level"], r["commission_rate"]))
    for _, r in enriched_tx[["channel", "status"]].drop_duplicates().iterrows():
        cur.execute("INSERT INTO dim_channel (channel_name, status) VALUES (?, ?)", (r["channel"], r["status"]))
    conn.commit()

    acc_map = dict(cur.execute("SELECT account_id, account_key FROM dim_account").fetchall())
    merch_map = dict(cur.execute("SELECT merchant_id, merchant_key FROM dim_merchant").fetchall())
    chan_map = {(cn, st): ck for ck, cn, st in cur.execute("SELECT channel_key, channel_name, status FROM dim_channel").fetchall()}

    fact_rows = []
    for _, r in enriched_tx.iterrows():
        dt_k = int(pd.to_datetime(r["trans_date"]).strftime("%Y%m%d"))
        fact_rows.append((
            r["transaction_id"], dt_k, acc_map[r["account_id"]], merch_map[r["merchant_id"]],
            chan_map[(r["channel"], r["status"])], r["amount"], r["fee_amount"], r["net_amount"], r["is_high_value"]
        ))
    cur.executemany("INSERT INTO fact_transactions (transaction_id, date_key, account_key, merchant_key, channel_key, amount, fee_amount, net_amount, is_high_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", fact_rows)

    cur.executescript("""
    CREATE VIEW v_monthly_financial_volume AS
    SELECT d.year_number, d.month_number, d.month_name, COUNT(f.trans_key) AS transaction_count, ROUND(SUM(f.amount), 2) AS gross_transaction_volume, ROUND(SUM(f.fee_amount), 2) AS fee_revenue, ROUND(AVG(f.amount), 2) AS avg_ticket_size
    FROM fact_transactions f JOIN dim_date d ON f.date_key = d.date_key JOIN dim_channel c ON f.channel_key = c.channel_key WHERE c.status = 'Settled' GROUP BY d.year_number, d.month_number, d.month_name ORDER BY d.year_number, d.month_number;

    CREATE VIEW v_merchant_category_metrics AS
    SELECT m.category, COUNT(f.trans_key) AS transaction_count, ROUND(SUM(f.amount), 2) AS total_category_volume, ROUND(SUM(f.fee_amount), 2) AS category_fee_revenue
    FROM fact_transactions f JOIN dim_merchant m ON f.merchant_key = m.merchant_key JOIN dim_channel c ON f.channel_key = c.channel_key WHERE c.status = 'Settled' GROUP BY m.category ORDER BY total_category_volume DESC;

    CREATE VIEW v_channel_distribution AS
    SELECT c.channel_name, COUNT(f.trans_key) AS transaction_count, ROUND(SUM(f.amount), 2) AS total_channel_volume
    FROM fact_transactions f JOIN dim_channel c ON f.channel_key = c.channel_key WHERE c.status = 'Settled' GROUP BY c.channel_name ORDER BY total_channel_volume DESC;
    """)
    conn.commit()
    print("  ✓ Star Schema & Analytical Views loaded.")

    # 5. DASHBOARD
    print("\n[STAGE 5] Rendering Executive Visual Dashboard...")
    df_monthly = pd.read_sql_query("SELECT * FROM v_monthly_financial_volume", conn)
    df_cat = pd.read_sql_query("SELECT * FROM v_merchant_category_metrics", conn)
    df_chan = pd.read_sql_query("SELECT * FROM v_channel_distribution", conn)
    df_risk = pd.read_sql_query("SELECT risk_tier, COUNT(*) AS account_count FROM dim_account GROUP BY risk_tier", conn)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("FINTECH TRANSACTION MONITORING & DATA WAREHOUSE - EXECUTIVE DASHBOARD\nArchitected by: Ayan Patel", fontsize=16, fontweight='bold', y=0.98)

    # 1
    ax1 = axes[0, 0]
    ax1.plot(df_monthly["month_name"], df_monthly["gross_transaction_volume"], marker='o', linewidth=2.5, color='#1f77b4', label="Gross Volume (INR)")
    ax1.plot(df_monthly["month_name"], df_monthly["fee_revenue"], marker='s', linewidth=2.5, color='#d62728', label="Fee Revenue (INR)")
    ax1.set_title("Monthly Transaction Volume & Fee Revenue Trajectory", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Volume (INR)", fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # 2
    ax2 = axes[0, 1]
    bars = ax2.barh(df_cat["category"], df_cat["total_category_volume"], color=['#2b5c8f', '#4682b4', '#5c9bd5', '#85b8e2', '#a6cde2'])
    ax2.set_title("Spend Volume by Merchant Category", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Volume (INR)", fontsize=10)
    for bar in bars:
        w = bar.get_width()
        ax2.text(w * 0.70, bar.get_y() + bar.get_height()/2, f"INR {w:,.0f}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # 3
    ax3 = axes[1, 0]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
    ax3.pie(df_chan["total_channel_volume"], labels=df_chan["channel_name"], autopct='%1.1f%%', startangle=140, colors=colors[:len(df_chan)], explode=[0.05]*len(df_chan), shadow=True)
    ax3.set_title("Transaction Channel Volume Share", fontsize=12, fontweight='bold')

    # 4
    ax4 = axes[1, 1]
    sns.barplot(data=df_risk, x="account_count", y="risk_tier", hue="risk_tier", legend=False, palette="mako", ax=ax4)
    ax4.set_title("Customer Account Distribution Across Risk Tiers", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Number of Accounts", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_path = os.path.join(REPORTS_DIR, "executive_dashboard.png")
    plt.savefig(dash_path, dpi=300)
    plt.close()
    conn.close()

    print(f"  ✓ Dashboard saved to: {dash_path}")
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY (AUTHOR: AYAN PATEL)")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
