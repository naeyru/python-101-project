import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the Bitcoin price data
bitcoin_df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# Convert Unix timestamps to datetime
bitcoin_df['Timestamp'] = pd.to_datetime(bitcoin_df['Timestamp'], unit='s')

# Drop rows with NaN values
bitcoin_df = bitcoin_df.dropna()

# Assuming the CSV file has columns 'created_at', 'user_followers_count', 'text', and 'Sentiment_Score'
sentiment_df = pd.read_csv('sorted_sentiments.csv', parse_dates=['created_at'])

# Filter out rows where the sentiment score is zero
filtered_sentiment_df = sentiment_df[sentiment_df['text'] != 0]

# Group by year and calculate the mean sentiment score for each year
yearly_sentiment_df = filtered_sentiment_df.groupby(filtered_sentiment_df['created_at'].dt.year).mean(numeric_only=True)

# Plotting both Bitcoin price and Yearly Mean Sentiment Analysis Results
plt.figure(figsize=(15, 8))

# Plot Bitcoin Price Over Time
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(bitcoin_df['Timestamp'], bitcoin_df['Close'], label='Bitcoin Close Price')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

# Plot Yearly Mean Sentiment Analysis Results
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.plot(yearly_sentiment_df.index, yearly_sentiment_df['text'], marker='o', label='Mean Sentiment Score')
plt.title('Yearly Mean Sentiment Analysis Results')
plt.xlabel('Year')
plt.ylabel('Mean Sentiment Score')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the overall plot with both subplots
plt.show()
