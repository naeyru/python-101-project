"""
Thomas Safago
11/11/2023
Data sorting (into one file)
"""


import pandas as pd


def main():
    data = pd.read_csv("new_data/sentiment/sentiments.csv", parse_dates=["created_at"], low_memory=False)
    data = data.sort_values(by="created_at")
    data.to_csv("test.csv", date_format="%Y %B %d", index=False)


if __name__ == "__main__":
    main()