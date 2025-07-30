import csv
import datetime

import pandas

from Jandas.dataframe import DataFrame
from Jandas.series import Series
from Jandas.vector import Vector


def read_csv(file_path: str) -> pandas.DataFrame:
    """
    Read a CSV file and return a DataFrame object.

    Parameters:
    - file_path (str): The path to the CSV file to read.

    Returns:
    - DataFrame: A DataFrame containing the data from the CSV file with column names.

    Notes:
    - Automatically attempts to convert numeric values to int or float.
    - If conversion fails, the value is left as a string.
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        # Try to read the first row (header)
        try:
            columns = next(reader)
        except StopIteration:
            # If the file is empty, return an empty DataFrame
            return DataFrame([], [])

        # If the header is empty, also return an empty DataFrame
        if not columns:
            return DataFrame([], [])
        data = []
        for row in reader:
            # Automatic type conversion
            parsedRow = []
            for value in row:
                try:
                    # Convert to float if possible
                    parsedValue = float(value)
                    if parsedValue.is_integer():
                        parsedValue = int(parsedValue)
                except ValueError:
                    # Leave as string if conversion fails
                    parsedValue = value
                parsedRow.append(parsedValue)
            data.append(parsedRow)
        return DataFrame(data, columns)


def merge(left, right, how='inner', left_on=None, right_on=None, left_index=False, right_index=False):
    """
    Merge two DataFrames based on indices or specific columns.

    Parameters:
    - left (DataFrame): The left DataFrame to merge.
    - right (DataFrame): The right DataFrame to merge.
    - how (str): Type of merge to perform. Options: 'inner', 'outer', 'left', 'right'. Default is 'inner'.
    - left_on (str): Column name in the left DataFrame to join on (required if `left_index` is False).
    - right_on (str): Column name in the right DataFrame to join on (required if `right_index` is False).
    - left_index (bool): If True, use the index from the left DataFrame as the join key. Default is False.
    - right_index (bool): If True, use the index from the right DataFrame as the join key. Default is False.

    Returns:
    - DataFrame: A new DataFrame containing the merged data.

    Raises:
    - ValueError: If `how` is not a valid merge type.
    """
    if left_index:
        leftKeys = left.index
    else:
        leftKeys = [row[left.columns.index(left_on)] for row in left.data]

    if right_index:
        rightKeys = right.index
    else:
        rightKeys = [row[right.columns.index(right_on)] for row in right.data]

    mergedData = []
    mergedColumns = left.columns + [col for col in right.columns if col not in left.columns]

    if how == 'inner':
        commonKeys = set(leftKeys).intersection(rightKeys)
    elif how == 'outer':
        commonKeys = set(leftKeys).union(rightKeys)
    elif how == 'left':
        commonKeys = set(leftKeys)
    elif how == 'right':
        commonKeys = set(rightKeys)
    else:
        raise ValueError("Invalid `how` value")

    for key in commonKeys:
        leftRows = [row for row, k in zip(left.data, leftKeys) if k == key]
        rightRows = [row for row, k in zip(right.data, rightKeys) if k == key]

        if not leftRows:
            leftRows = [Vector([None] * len(left.columns))]
        if not rightRows:
            rightRows = [Vector([None] * len(right.columns))]

        for lRow in leftRows:
            for rRow in rightRows:
                mergedRow = list(lRow) + [r for r, col in zip(rRow, right.columns) if col not in left.columns]
                mergedData.append(mergedRow)

    return DataFrame(data=mergedData, columns=mergedColumns)


def concat(dfs, axis=0):
    """
    Concatenate multiple DataFrames along a specified axis.

    Args:
        dfs (list): List of DataFrames to concatenate.
        axis (int): The axis along which to concatenate (0 for rows, 1 for columns).

    Returns:
        DataFrame: The concatenated DataFrame.

    Raises:
        ValueError: If the axis is invalid or the DataFrames are incompatible.
    """
    if not dfs:
        raise ValueError("No DataFrames to concatenate")

    if axis not in (0, 1):
        raise ValueError("Invalid axis value. Axis must be 0 or 1.")

    # Validate all are DataFrames and have consistent columns/rows as needed
    if axis == 0:
        # Vertical concat: columns must be the same (or compatible)
        baseColumns = dfs[0].columns
        for df in dfs[1:]:
            if df.columns != baseColumns:
                raise ValueError("All DataFrames must have the same columns to concatenate along axis 0")
        # Combine all data rows
        combinedData = []
        for df in dfs:
            combinedData.extend(df.data)
        return DataFrame(combinedData, dfs[0].columns)

    else:  # axis == 1, horizontal concat: number of rows must be equal
        baseLen = len(dfs[0].data)
        for df in dfs[1:]:
            if len(df.data) != baseLen:
                raise ValueError("All DataFrames must have the same number of rows to concatenate along axis 1")
        concatenatedData = []
        for rows in zip(*(df.data for df in dfs)):
            newRow = []
            for row in rows:
                newRow.extend(row)
            concatenatedData.append(newRow)
        # Combine columns from all dfs
        combinedColumns = dataframe.JIndex([])
        for df in dfs:
            combinedColumns += df.columns
        return DataFrame(concatenatedData, combinedColumns)


def to_numeric(val, errors='raise'):
    """
    Convert a value or a Series of values to float or int.

    If given a Series, return a Series of numeric values.
    If given a scalar, return a single numeric value.

    Args:
        val (Series | str | float | int): Input value(s).
        errors (str): 'raise' or 'coerce'.

    Returns:
        Series or float or int: Converted numeric value(s).
    """

    def convert(v):
        try:
            return float(v) if '.' in str(v) else int(v)
        except Exception:
            if errors == 'coerce':
                return None
            else:
                raise ValueError("Cannot convert value: {}".format(v))

    if isinstance(val, Series):
        return Series([convert(v) for v in val.data])
    else:
        return convert(val)


def arrange(start, stop=None, step=1):
    """
    Mimics numpy.arange() using pure Python.

    Args:
        start (int): Start value, or stop if stop is None.
        stop (int, optional): Stop value (non-inclusive).
        step (int, optional): Step size.

    Returns:
        list: A list of numbers from start to stop (exclusive) by step.
    """
    if stop is None:
        start, stop = 0, start

    if step == 0:
        raise ValueError("step must not be zero")

    result = []
    i = start
    if step > 0:
        while i < stop:
            result.append(i)
            i += step
    else:
        while i > stop:
            result.append(i)
            i += step
    return result


def to_datetime(obj, fmt=None, errors="raise", default_year=None):
    """
    Convert a string or Series of strings to datetime objects with support for multiple formats.

    Args:
        obj (str | Series): A datetime string or a Series of strings.
        fmt (str | list | None): A format string, list of formats, or None to try common formats.
        errors (str): 'raise' or 'coerce'.
        default_year (int or None): If a date has no year (e.g., "March 3"), use this year.

    Returns:
        datetime.datetime or Series: Parsed datetime(s)
    """
    defaultFormats = [
        "%Y/%m/%d",  # 2024/01/01
        "%Y-%m-%d",  # 2024-01-01
        "%d-%m-%Y",  # 01-02-2024
        "%m-%d-%Y",  # 01-02-2024 (ambiguous)
        "%B %d",  # March 3
        "%b %d",  # Mar 3
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d",
    ]

    if fmt is None:
        formats = defaultFormats
    elif isinstance(fmt, str):
        formats = [fmt]
    elif isinstance(fmt, list):
        formats = fmt
    else:
        raise TypeError("fmt must be a string, list of strings, or None")

    def parse(val):
        val = val.strip()
        for f in formats:
            try:
                parsed = datetime.datetime.strptime(val, f)
                # If the format didn’t include a year, inject it
                if "%Y" not in f and default_year is not None:
                    parsed = parsed.replace(year=default_year)
                return parsed
            except Exception:
                continue
        if errors == "coerce":
            return None
        else:
            raise ValueError("Cannot parse value to datetime: '{}'".format(val))

    if isinstance(obj, Series):
        return Series([parse(x) for x in obj.data])
    else:
        return parse(obj)

def from_dataset(dataset):
    """
    Convert an Ignition Dataset to a Jandas DataFrame.

    Parameters:
    - dataset (Dataset): Ignition Dataset object.

    Returns:
    - DataFrame: Jandas DataFrame with the same data and column names.
    """
    if system is None:
        raise RuntimeError("from_dataset requires Ignition's system module.")

    columnNames = list(dataset.getColumnNames())
    data = []

    for rowIndex in range(dataset.getRowCount()):
        row = []
        for col in columnNames:
            row.append(dataset.getValueAt(rowIndex, col))
        data.append(row)

    return DataFrame(data=data, columns=columnNames)


def to_dataset(df):
    """
    Convert a Jandas DataFrame to an Ignition Dataset.

    Parameters:
    - df (DataFrame): Jandas DataFrame to convert.

    Returns:
    - Dataset: Ignition Dataset with the same data and column names.
    """
    if system is None:
        raise RuntimeError("to_dataset requires Ignition's system module.")

    headers = df.columns
    data = [list(row) for row in df.data]
    return system.dataset.toDataSet(headers, data)
