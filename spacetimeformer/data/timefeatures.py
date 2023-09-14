import pandas as pd
import numpy as np


def time_features(dates, main_df=None):
    if main_df is None:
        main_df = pd.DataFrame({})
    years = dates.apply(lambda row: row.year)
    max_year = years.max()
    min_year = years.min()
    main_df["Year"] = dates.apply(
        lambda row: (row.year - min_year) / max(1.0, (max_year - min_year))
    )

    main_df["Month"] = dates.apply(lambda row: 2.0 * ((row.month - 1) / 11.0) - 1.0, 1)
    main_df["Day"] = dates.apply(lambda row: 2.0 * ((row.day - 1) / 30.0) - 1.0, 1)
    main_df["Weekday"] = dates.apply(lambda row: 2.0 * (row.weekday() / 6.0) - 1.0, 1)
    main_df["Hour"] = dates.apply(lambda row: 2.0 * ((row.hour) / 23.0) - 1.0, 1)
    main_df["Minute"] = dates.apply(lambda row: 2.0 * ((row.minute) / 59.0) - 1.0, 1)
    main_df["Datetime"] = dates
    return main_df


def cycle_features(cycles, main_df=None):
    if main_df is None:
        main_df = pd.DataFrame({})

    main_df["cycles1"] = cycles.apply(lambda row: np.sin(row/(1*4*np.pi)), 1)
    main_df["cycles10"] = cycles.apply(lambda row: np.sin(row/(10*4*np.pi)), 1)
    main_df["cycles100"] = cycles.apply(lambda row: np.sin(row/(100*4*np.pi)), 1)
    return main_df
