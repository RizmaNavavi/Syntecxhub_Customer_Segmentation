"""
===========================================
Syntechhub Internship - Project 1
Customer Segmentation using RFM Analysis
===========================================
"""

# ---------- STEP 1: Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os

print("✅ Step 1: Libraries imported successfully\n")

# Make sure the data folder exists
os.makedirs('data', exist_ok=True)

# ---------- STEP 2: Load Dataset ----------
df = pd.read_csv('data/online_retail.csv', encoding='ISO-8859-1')
print(f"✅ Step 2: Dataset loaded — Shape: {df.shape}\n")

# ---------- STEP 3: Explore the Data ----------
print("--- Data Info ---")
df.info()
print("\n--- Missing Values ---")
print(df.isnull().sum())
print()

# ---------- STEP 4: Clean the Data ----------
print(f"Before cleaning: {df.shape}")

# Remove missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Remove canceled orders
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative / zero Quantity and UnitPrice
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"After cleaning: {df.shape}\n")

# ---------- STEP 5: Calculate RFM Metrics ----------
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
print(f"Snapshot Date: {snapshot_date}")

# Create TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Group by customer
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                     # Frequency
    'TotalPrice': 'sum'                                         # Monetary
})

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

print(f"Total customers: {len(rfm)}\n")
print(rfm.head())
print()

# ---------- STEP 6: Create RFM Scores ----------
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print("✅ Step 6: RFM scores created\n")

# ---------- STEP 7: Segment Customers ----------
def segment_customer(row):
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
    if r >= 3 and f >= 3 and m >= 3:
        return 'Loyal'
    elif r <= 2 and f >= 3:
        return 'Churn Risk'
    elif r >= 3 and f <= 2:
        return 'New'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("--- Customers per Segment ---")
print(rfm['Segment'].value_counts())
print()

# ---------- STEP 8: Analyze Segments ----------
segment_analysis = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)

print("--- Segment Analysis ---")
print(segment_analysis)
print()

# ---------- STEP 9: Bar Chart ----------
plt.figure(figsize=(10, 5))
sns.countplot(data=rfm, x='Segment',
              order=rfm['Segment'].value_counts().index,
              palette='viridis')
plt.title('Number of Customers per Segment', fontsize=14)
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('data/segment_count.png', dpi=150)
plt.close()
print("✅ Step 9: Bar chart saved → data/segment_count.png")

# ---------- STEP 10: Scatter Plot ----------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Frequency', y='Monetary',
                hue='Segment', palette='Set2', alpha=0.7)
plt.title('Customer Segments: Frequency vs Monetary', fontsize=14)
plt.xlabel('Frequency (Number of Orders)')
plt.ylabel('Monetary (Total Spend)')
plt.tight_layout()
plt.savefig('data/scatter_plot.png', dpi=150)
plt.close()
print("✅ Step 10: Scatter plot saved → data/scatter_plot.png")

# ---------- STEP 11: Save Results ----------
rfm.to_csv('data/rfm_segments.csv')
print("✅ Step 11: RFM table saved → data/rfm_segments.csv\n")

print("=" * 50)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print(f"   Total customers segmented: {len(rfm)}")
print("=" * 50)