"""
Thomas Safago
11/10/2023
Python file to clean data and make it into a more digestible, workable format for sentiment analysis/date sorting.
"""


import pandas as pd


def data_testing():  # Function to test data manip at a smaller scale before working with large csv files.
    data = pd.read_csv("original_data/testdata.csv", parse_dates=["created_at"])  # Will throw warning, ok
    new_data = data[["created_at", "user_followers_count", "text"]]
    new_data = new_data.sort_values(by="created_at")
    new_data.to_csv("original_data/newtestdata.csv", date_format="%Y %B %d", index=False)  # Year, short month, day


def clean_data(*args):  # Args should be filepaths
    for filepath in args:
        data = pd.read_csv("original_data/" + filepath, parse_dates=["created_at"], low_memory=False)[["created_at", "user_followers_count", "text"]]
        data = data.sort_values(by="created_at")  # Sort by date, good
        data.to_csv("new_data/" + filepath, date_format="%Y %B %d", index=False)
        print(f"<{filepath}> done!")

def main():
    clean_data("BTCP2.csv", "BTCP3.csv", "BTCP4.csv", "BTCP5.csv", "BTCP6.csv", "BTCP7.csv", "BTCP8.csv")


if __name__ == "__main__":
    main()
