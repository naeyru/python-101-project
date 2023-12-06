"""
Thomas Safago
Graph for follower/sentiment heuristic.
"""


import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd


def compile_heuristic():
    data = pd.read_csv("sorted_sentiments.csv", parse_dates=["created_at"], low_memory=False)
    # new_data = data[["created_at", "follower_sentiment_heuristic"]]

    t_data = {
        "created_at": data["created_at"],
        "heuristic": data["text"] * data["user_followers_count"]
    }

    new_data = pd.DataFrame(t_data)

    daily_records = new_data.groupby(new_data["created_at"]).count()["heuristic"]
    daily_total_heuristic = new_data.groupby(new_data["created_at"]).sum()["heuristic"]

    # print(daily_records, "\n\n", daily_total_heuristic)

    n_data = daily_total_heuristic / daily_records

    n_data.to_csv("daily_heuristic_average.csv", date_format="%Y %B %d", index=True)


def compile_heuristic_2():
    data = pd.read_csv("sorted_sentiments.csv", parse_dates=["created_at"], low_memory=False)
    data = data[data["text"] != 0]

    t_data = {
        "created_at": data["created_at"],
        "heuristic": data["text"] * data["user_followers_count"]
    }

    new_data = pd.DataFrame(t_data)

    daily_records = new_data.groupby(new_data["created_at"]).count()["heuristic"]
    daily_total_heuristic = new_data.groupby(new_data["created_at"]).sum()["heuristic"]

    # print(daily_records, "\n\n", daily_total_heuristic)

    n_data = daily_total_heuristic / daily_records

    n_data.to_csv("daily_heuristic_average_2.csv", date_format="%Y %B %d", index=True)
    print(n_data)


def compile_user_follower_heuristic():
    data = pd.read_csv("sorted_sentiments.csv", parse_dates=["created_at"], low_memory=False)
    # new_data = data[["created_at", "follower_sentiment_heuristic"]]

    t_data = {
        "created_at": data["created_at"],
        "heuristic": data["user_followers_count"]
    }

    new_data = pd.DataFrame(t_data)

    daily_records = new_data.groupby(new_data["created_at"]).count()["heuristic"]
    daily_total_heuristic = new_data.groupby(new_data["created_at"]).sum()["heuristic"]

    # print(daily_records, "\n\n", daily_total_heuristic)

    n_data = daily_total_heuristic / daily_records

    n_data.to_csv("daily_follower_heuristic_average.csv", date_format="%Y %B %d", index=True)


def main():
    compile_heuristic()
    heur_data = pd.read_csv("daily_heuristic_average.csv", parse_dates=["created_at"], low_memory=False)
    heur_data = heur_data[(heur_data['created_at'] > '2013-9-30') & (heur_data['created_at'] <= '2017-8-11')]

    btc_data = pd.read_csv("btc.csv", parse_dates=["Date"], low_memory=False)

    btc_needed = pd.DataFrame({
        "Date": btc_data["Date"],
        "Price (USD)": btc_data["24h High (USD)"]
    })

    btc_needed = btc_needed[(btc_needed['Date'] > '2013-9-30') & (btc_needed['Date'] <= '2017-8-11')]
    btc_needed = btc_needed.set_index(btc_needed.columns[0])
    btc_needed.plot()

    dates = heur_data["created_at"].tolist()
    filtered_values = savgol_filter(heur_data["heuristic"], 420, 1, mode="interp")

    plt.plot(dates, filtered_values)
    plt.gcf().autofmt_xdate()
    plt.ylim(-1000, 4000)
    plt.savefig("3_follower_data.png")

    # Turn results into interactive model to estimate closing prices of data for demo
    data_map = {str(dates[i])[:-9]:filtered_values.tolist()[i] for i in range(len(dates))}
    btc_map = {str(dates[i])[:-9]:btc_needed["Price (USD)"].tolist()[i] for i in range(len(dates))}
    demo = True

    print(f"Welcome to the demo! Enter a date between {list(data_map.keys())[0]} and {list(data_map.keys())[-1]}.\n"
          f"This will give you the model's estimated value and compare it to the real value of BTC at that date.\n"
          f"Type 'q' to quit, and type 'i' to see more information.\n")

    while demo:
        date = input("[YYYY-MM-DD]: ")
        if date in data_map.keys():
            print(f"Model prediction for {date}: ${round(data_map[date], 2)}\n"
                  f"BTC 24-hour high for {date}: ${round(btc_map[date], 2)}\n"
                  f"Error: {abs(round(((data_map[date] - btc_map[date])/(abs(data_map[date] + btc_map[date])/2))*100, 2))}%\n")
        elif date == 'q':
            print("Quitting program.")
            demo = False
        elif date == 'i':
            avg_err = 0
            for d in data_map.keys():
                avg_err += abs(round(((data_map[d] - btc_map[d]) / (abs(data_map[d] + btc_map[d]) / 2)) * 100, 2))
            print(f"Average error: {round(avg_err/float(len(data_map)), 2)}%")
            print(f"Total entries: {len(data_map)}\n")
        else:
            print("Invalid date/input!")


if __name__ == "__main__":
    main()
