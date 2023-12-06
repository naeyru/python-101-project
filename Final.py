import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the Bitcoin price data
df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# Convert Unix timestamps to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Drop rows with NaN values
df = df.dropna()

# Extract year from Timestamp
df['Year'] = df['Timestamp'].dt.year

# Monthly Mean Sentiment Analysis Results
sentiment_df = pd.read_csv('sorted_sentiments.csv', parse_dates=['created_at'])
filtered_sentiment_df = sentiment_df[sentiment_df['text'] != 0]

# Set timezone for sentiment_df index
monthly_sentiment_df = filtered_sentiment_df.groupby([filtered_sentiment_df['created_at'].dt.to_period("M")]).mean(numeric_only=True)
monthly_sentiment_df.index = pd.to_datetime(monthly_sentiment_df.index.astype(str)).tz_localize('UTC')

# Set timezone for sentiment_df 'created_at' column
filtered_sentiment_df['created_at'] = filtered_sentiment_df['created_at'].dt.tz_localize('UTC')
filtered_sentiment_df['Year'] = filtered_sentiment_df['created_at'].dt.year

# Plotting for each year
for year in df['Year'].unique():
    # Filter data for the current year
    year_df = df[df['Year'] == year]

    # Plot Bitcoin Price Over Time
    plt.figure(figsize=(15, 8))
    ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    ax1.plot(year_df['Timestamp'], year_df['Close'], label='Close Price', color='blue')
    ax1.set_title(f'Bitcoin Price - {year}')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Close Price')
    ax1.legend()
    ax1.grid(True)

    # Filter sentiment data for the current year
    year_sentiment_df = monthly_sentiment_df[monthly_sentiment_df.index.year == year]

    # Plot Monthly Mean Sentiment Analysis Results with abbreviated Month-Year format
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # 2 rows, 1 column, second subplot
    ax2.plot(year_sentiment_df.index, year_sentiment_df['text'], marker='o', label='Mean Sentiment Score', color='orange')

    # Set the date formatter for the x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

    ax2.set_xlabel('Timestamp / Month-Year')
    ax2.set_ylabel('Mean Sentiment Score')
    ax2.set_title(f'Monthly Mean Sentiment Analysis - {year}')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show the plot
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()
