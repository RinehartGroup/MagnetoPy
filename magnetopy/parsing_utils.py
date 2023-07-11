import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def label_clusters(
    vals: pd.Series, eps: float = 0.001, min_samples: int = 10
) -> np.ndarray:
    """For determining the nominal values of data in a series containing one or more
    nominal values with some noise.

    Parameters
    ----------
    vals : pd.Series
        A series of data containing one or more nominal values with some noise.
    eps : float, optional
        Passed to `sklearn.cluster.DBSCAN()`. The maximum distance between two samples
        for one to be considered as in the neighborhood of the other, by default 0.001.
    min_samples : int, optional
        Passed to `sklearn.cluster.DBSCAN()`. The number of samples in a neighborhood
        for a point to be considered as a core point, by default 10.

    Returns
    -------
    np.ndarray
        An array of the same size as `vals` which contains the cluster labels for each
        element in `vals`. Noisy samples are given the label -1. A `vals` series
        containing, for example, one nominal temperature with noise should return an
        array with only one cluster label of -1.

    """
    reshaped_vals = vals.values.reshape(-1, 1)
    scaler = StandardScaler()
    reshaped_normalized_vals = scaler.fit_transform(reshaped_vals)
    # adjust eps based on the range of the normalized data
    eps_scaled = np.ptp(reshaped_normalized_vals) * eps
    dbscan = DBSCAN(eps=eps_scaled, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(reshaped_normalized_vals)
    # all values should be assigned to a cluster
    # if not, increase eps and try again
    if -1 in cluster_labels:
        cluster_labels = label_clusters(vals, eps=eps * 10, min_samples=min_samples)
    return cluster_labels


def unique_values(
    x: pd.Series, eps: float = 0.001, min_samples: int = 10, ndigits: int = 0
) -> list[int | float]:
    """Given a series of data containing one or more nominal values with some noise,
    returns a list of the nominal values.

    Parameters
    ----------
    x : pd.Series
        A series of data containing one or more nominal values with some noise.
    eps : float, optional
        Passed to `sklearn.cluster.DBSCAN()`. The maximum distance between two samples
        for one to be considered as in the neighborhood of the other, by default 0.001.
    min_samples : int, optional
        Passed to `sklearn.cluster.DBSCAN()`. The number of samples in a neighborhood
        for a point to be considered as a core point, by default 10.
    ndigits : int, optional
        The number of digits after the decimal point to round the nominal values to,
        by default 0.

    Returns
    -------
    list[float]
        The nominal values in `x` with the noise removed.
    """
    cluster_labels = label_clusters(x, eps=eps, min_samples=min_samples)
    unique_vals = []
    for i in np.unique(cluster_labels):
        # average the values in each cluster
        unique_val = np.mean(x[cluster_labels == i])
        unique_val = round(unique_val, ndigits)
        if ndigits == 0:
            unique_val = int(unique_val)
        unique_vals.append(unique_val)
    return unique_vals


def find_outlier_indices(x: pd.Series, threshold: float = 3) -> list[int]:
    """Finds the indices of outliers in a series of data.

    Parameters
    ----------
    x : pd.Series
        A series of data.
    threshold : float, optional
        The number of standard deviations from the mean to consider a value an outlier,
        by default 3.

    Returns
    -------
    list[int]
        The indices of the outliers in `x`.
    """
    z_scores = (x - x.mean()) / x.std()
    outliers = z_scores.abs() > threshold
    return list(outliers[outliers].index)


def find_temp_turnaround_point(df: pd.DataFrame) -> int:
    """Finds the index of the temperature turnaround point in a dataframe of
    a ZFCFC experiment which includes a column "Temperature (K)". Can handle two cases
    in which a single dataframe contains first a ZFC experiment, then a FC experiment:
    - Case 1: ZFC temperature monotonically increases, then FC temperature
    monotonically decreases.
    - Case 2: ZFC temperature monotonically increases, the temperature is reset to a
    lower value, then FC temperature monotonically increases.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of a ZFCFC experiment which includes a column "Temperature (K)".

    Returns
    -------
    int
        The index of the temperature turnaround point.

    """
    outlier_indices = find_outlier_indices(df["Temperature (K)"].diff())
    if len(outlier_indices) == 0:
        # zfc temp increases, fc temp decreases
        zero_point = abs(df["Temperature (K)"].iloc[20:-20].diff()).idxmin()
        return zero_point
    else:
        # zfc temp increases, reset temp, fc temp increases
        return outlier_indices[0]


def find_sequence_starts(x: pd.Series, flucuation_tolerance: float = 0) -> list[int]:
    """Find the indices of the start of each sequence in a series of data,
    where a sequences is defined as a series of numbers that constantly increase or decrease.
    Changes below `fluctuation_tolerance` are ignored.

    Example:
    ```
    >>>x = pd.Series([0, 1, 2, 3, 4, 3, 2, 1])
    >>>_find_sequence_starts(x)
    [0, 5]

    >>>y = pd.Series([0, 1, 2, 3, 0, 1, 2, 3])
    >>>_find_sequence_starts(y)
    [0, 4]
    ```
    """
    if flucuation_tolerance < 0:
        raise ValueError("fluctuation_tolerance must be non-negative")
    df = pd.DataFrame({"x": x, "diff": x.diff()})
    df["direction"] = np.where(df["diff"] > 0, 1, -1)
    start: int = df.index.start  # type: ignore
    df.at[start, "direction"] = df.at[
        start + 1, "direction"
    ]  # since the first value of diff is NaN
    # if there's a really small diff value with the opposite sign of diff values around it, it's probably a mistake
    sequence_starts = [0]
    for i in df[start + 2 :].index:
        last2_dir = df.at[i - 2, "direction"]  # type: ignore
        last1_dir = df.at[i - 1, "direction"]  # type: ignore
        current_dir = df.at[i, "direction"]
        try:
            next1_dir = df.at[i + 1, "direction"]  # type: ignore
            next2_dir = df.at[i + 2, "direction"]  # type: ignore
            next3_dir = df.at[i + 3, "direction"]  # type: ignore
        except KeyError:
            # reached end of dataframe
            break

        # below handles, for example, zfc from 5 to 300 K, drop temp to 5 K, fc from 5 to 300 K
        if (current_dir != last1_dir) and (current_dir != next1_dir):
            if abs(df.at[i, "diff"]) < flucuation_tolerance:
                # this is a fluctuation and should be ignored
                df.at[i, "direction"] = last1_dir
                current_dir = last1_dir
            sequence_starts.append(i)  # type: ignore

        # below handles, for example, zfc from 5 to 300 K, fc from 300 to 5 K
        # assumes there won't be any fluctuations at the beginning of a sequence
        if (
            (last2_dir == last1_dir)
            and (current_dir != last1_dir)
            and (current_dir == next1_dir == next2_dir == next3_dir)
        ):
            sequence_starts.append(i)  # type: ignore

    return sequence_starts
