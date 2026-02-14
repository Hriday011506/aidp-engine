import pandas as pd

def preprocess_data():
    df = pd.read_csv("data/raw/retail_sales.csv")

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    monthly = df.groupby('month').agg({
        'sales': 'sum',
        'onpromotion': 'sum'
    }).reset_index()

    monthly.rename(columns={'sales': 'monthly_sales'}, inplace=True)

    monthly.to_csv("data/processed/monthly_data.csv", index=False)
