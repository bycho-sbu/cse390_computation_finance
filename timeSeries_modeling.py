import pandas as pd
import statsmodels.api as sm

def static_split(file_path: str, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time series dataset stored in a CSV file into training and testing sets based on the specified fraction for
    the training set.
    """
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be between 0 and 1.")

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    # take the last column as the time series
    df = df.iloc[:, -1].to_frame(name="value")

    split_idx = int(len(df) * train_frac)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()

    return train_data, test_data


def rolling_window_splits(file_path: str, train_size: int, test_size: int, slide_size: int = 1) -> list[tuple[int, int, int, int]]:
    """
    Generates index references for training and test sets using a rolling window approach for time series data.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive integers.")

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    df = df.iloc[:, -1].to_frame(name="value")
    n = len(df)

    index_references = []
    start_train = 0
    while start_train + train_size + test_size <= n:
        end_train = start_train + train_size
        start_test = end_train
        end_test = start_test + test_size
        index_references.append((start_train, end_train, start_test, end_test))
        start_train += slide_size

    return index_references


def expanding_window_splits(file_path: str, initial_train_size: int, test_size: int, expansion_step: int = 1) -> list[tuple[int, int, int, int]]:
    """
    Generates index references for training and test sets using an expanding window approach for time series data.
    """
    if initial_train_size <= 0 or test_size <= 0 or expansion_step <= 0:
        raise ValueError("all size parameters must be positive integers.")

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    df = df.iloc[:, -1].to_frame(name="value")
    n = len(df)

    index_references = []
    train_end = initial_train_size
    while train_end + test_size <= n:
        start_train = 0
        end_train = train_end
        start_test = end_train
        end_test = start_test + test_size
        index_references.append((start_train, end_train, start_test, end_test))
        train_end += expansion_step

    return index_references

def fit_ar1_model(train_data: pd.DataFrame):
    """
    Fits an AR(1) model to the provided time series training data.

    Returns:
    - statsmodels.tsa.arima.model.ARIMAResults: fitted AR(1) model object
    """
    import statsmodels.api as sm

    series = train_data.iloc[:, 0]  # assume only one column
    model = sm.tsa.arima.model.ARIMA(series, order=(1, 0, 0))
    ar1_model = model.fit()
    return ar1_model

import numpy as np
import pandas as pd
import statsmodels.api as sm

def ar1_mse_7030(file_path: str, train_frac: float = 0.7) -> tuple[float, float, sm.tsa.arima.model.ARIMAResults]:
    """
    runs a 70/30 (default) static split, fits ar(1) on the training set, and returns (train_mse, test_mse, fitted_model).

    parameters
    ----------
    file_path : str
        path to the whitespace-delimited file (the time series is the last column).
    train_frac : float
        fraction for training size (default 0.7).

    returns
    -------
    train_mse : float
    test_mse  : float
    model     : statsmodels.tsa.arima.model.ARIMAResults
    """
    # 1) split the data (your existing function)
    train_df, test_df = static_split(file_path, train_frac=train_frac)

    # 2) fit ar(1) on the training data (your existing function)
    model = fit_ar1_model(train_df)

    # 3) training mse
    # statsmodels aligns fittedvalues to the endogenous index and typically starts with a nan
    train_true = train_df.iloc[:, 0]
    train_fitted = model.fittedvalues
    # align and drop nans
    train_aligned = train_true.loc[train_fitted.index]
    mask = ~np.isnan(train_fitted.values)
    train_mse = float(np.mean((train_aligned.values[mask] - train_fitted.values[mask]) ** 2))

    # 4) test mse: forecast len(test) steps and compare
    steps = len(test_df)
    fc = model.get_forecast(steps=steps)
    test_pred = fc.predicted_mean.values
    test_true = test_df.iloc[:, 0].values
    test_mse = float(np.mean((test_true - test_pred) ** 2))

    return train_mse, test_mse, model

if __name__ == "__main__":
    # Example usage
    file_path = "path/to/your/data.txt"
    train_mse, test_mse, model = ar1_mse_7030(file_path)
    print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
    print(model.summary())