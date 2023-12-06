import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the Bitcoin price data
bitcoin_df = pd.read_csv('BTC_USD_2013-09-30_2021-03-29-CoinDesk (2).csv')

# Convert the 'Date' column to datetime format
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])

# Drop rows with NaN values
bitcoin_df = bitcoin_df.dropna()

# Assuming the CSV file has columns 'Date', 'Closing Price (USD)', '24h Open (USD)', '24h High (USD)', '24h Low (USD)'
# Replace the column names accordingly
bitcoin_df.rename(columns={
    'Date': 'Timestamp',
    'Closing Price (USD)': 'Close',
    '24h Open (USD)': 'Open',
    '24h High (USD)': 'High',
    '24h Low (USD)': 'Low'
}, inplace=True)

# Assuming the CSV file has columns 'created_at', 'user_followers_count', 'text', and 'Sentiment_Score'
sentiment_df = pd.read_csv('sorted_sentiments.csv', parse_dates=['created_at'])

# Filter out rows where the sentiment score is zero
filtered_sentiment_df = sentiment_df[sentiment_df['text'] != 0]

# Group by year and month and calculate the mean sentiment score for each month
monthly_sentiment_df = filtered_sentiment_df.groupby(
    [filtered_sentiment_df['created_at'].dt.year, filtered_sentiment_df['created_at'].dt.month]).mean(numeric_only=True)

# Plotting Bitcoin price and Monthly Mean Sentiment Analysis Results
years = bitcoin_df['Timestamp'].dt.year.unique()

for year in years:
    # Filter data for the current year
    year_bitcoin_df = bitcoin_df[bitcoin_df['Timestamp'].dt.year == year]

    # Filter monthly sentiment data for the current year
    monthly_sentiment_year_df = monthly_sentiment_df.loc[year]

    # Plot Bitcoin Price Over Time
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(year_bitcoin_df['Timestamp'], year_bitcoin_df['Close'], label='Bitcoin Close Price')
    plt.title(f'Bitcoin Price Over Time - {year}')
    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)

    # Plot Monthly Mean Sentiment Analysis Results
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(monthly_sentiment_year_df.index, monthly_sentiment_year_df['text'], marker='o',
             label='Mean Sentiment Score (Monthly)')
    plt.title(f'Monthly Mean Sentiment Analysis Results - {year}')
    plt.xlabel('Month')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the plot for the current year
    plt.show()
