import datetime
from collections import defaultdict

import Jandas


def _safe_compare(val, other, op):
    try:
        if val in [None, '']:
            return False
        return op(val, other)
    except Exception:
        return False


class Series:
    def __init__(self, data, index=None, name=None):
        if isinstance(data, str):
            raise ValueError("Series data should be a list, not a string")
        if isinstance(data, Jandas.vector.Vector):
            self.data = data.data
        else:
            self.data = list(data)
        self.index = index
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]

        elif isinstance(key, slice):
            # Slice by position
            return Series(self.data[key], name=self.name, index=self.index[key])

        elif isinstance(key, str):
            # Single label
            try:
                idx = self.index.index(key)
                return self.data[idx]
            except ValueError:
                raise KeyError("Label '{}' not found in index".format(key))

        elif isinstance(key, list):
            # List of labels or boolean mask
            if all(isinstance(k, bool) for k in key):
                filteredData = [d for d, mask in zip(self.data, key) if mask]
                filteredIndex = [i for i, mask in zip(self.index, key) if mask]
                return Series(filteredData, name=self.name, index=filteredIndex)

            elif all(isinstance(k, str) for k in key):
                indices = [self.index.index(k) for k in key]
                filteredData = [self.data[i] for i in indices]
                return Series(filteredData, name=self.name, index=key)

            else:
                raise TypeError("List indexing only supports labels or boolean masks.")

        else:
            raise TypeError("Unsupported index type: {}".format(type(key)))

    def __eq__(self, other):
        import operator
        if isinstance(other, Series):
            return Series(
                [_safe_compare(a, b, operator.eq) for a, b in zip(self.data, other.data)],
                name="({} == {})".format(self.name, other.name),
                index=self.index
            )
        else:
            return Series(
                [_safe_compare(val, other, operator.eq) for val in self.data],
                name="{} == {}".format(self.name, other),
                index=self.index
            )

    def __gt__(self, other):
        import operator
        if isinstance(other, (int, float)):
            return Series(
                [_safe_compare(val, other, operator.gt) for val in self.data],
                name="{} > {}".format(self.name, other),
                index=self.index
            )
        elif isinstance(other, Series):
            return Series(
                [_safe_compare(a, b, operator.gt) for a, b in zip(self.data, other.data)],
                name="({} > {})".format(self.name, other.name),
                index=self.index
            )
        else:
            raise TypeError("Unsupported operand type(s) for >: '{}' and '{}'".format(type(self), type(other)))

    def __lt__(self, other):
        import operator
        if isinstance(other, (int, float)):
            return Series(
                [_safe_compare(val, other, operator.lt) for val in self.data],
                name="{} < {}".format(self.name, other),
                index=self.index
            )
        elif isinstance(other, Series):
            return Series(
                [_safe_compare(a, b, operator.lt) for a, b in zip(self.data, other.data)],
                name="({} < {})".format(self.name, other.name),
                index=self.index
            )
        else:
            raise TypeError("Unsupported operand type(s) for <: '{}' and '{}'".format(type(self), type(other)))

    def __repr__(self):
        if self.data is None:
            return "Empty Series"

        if self.index is None:
            # Fall back to range index if none was provided
            self.index = list(range(len(self.data)))

        preview = '\n'.join(
            "{}    {}".format(idx, val) for idx, val in zip(self.index, self.data)
        )

        dtype = type(self.data[0]).__name__ if self.data else "object"
        return "{}\nName: {}, dtype: {}".format(preview, self.name, dtype)

    def __and__(self, other):
        return Series([a and b for a, b in zip(self.data, other.data)])

    def __or__(self, other):
        return Series([a or b for a, b in zip(self.data, other.data)])

    def __invert__(self):
        return Series([not x for x in self.data])

    def __len__(self):
        return len(self.data)

    def iloc(self, index):
        return self.data[index]

    @property
    def values(self):
        return self.data

    @property
    def kind(self):
        # Determine the kind of data in the series by checking the type of the first element
        if len(self.data) == 0:
            return None  # If the series is empty, return None

        # Check the type of the first element and return the kind accordingly
        firstValue = self.data[0]

        if isinstance(firstValue, (int, float)):
            if isinstance(firstValue, float):
                return 'float'
            else:
                return 'int'
        else:
            return 'object'

    def value_counts(self):
        """
        Returns a DataFrame with the counts of unique values.
        """
        from .dataframe import DataFrame
        counts = {}
        for value in self.data:
            counts[value] = counts.get(value, 0) + 1
        sortedCounts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Convert to the correct format: list of rows
        data = [[val, count] for val, count in sortedCounts]

        # Create the DataFrame with the structured data
        return DataFrame(data=data, columns=["Value", "Count"])

    def resample(self, rule):
        """
        Resample time-series data based on a frequency rule.

        Parameters:
        - rule (str): Resampling frequency ('D' = daily, 'M' = monthly, etc.)

        Returns:
        - Resampler: An object supporting aggregation like .mean()
        """
        if not all(isinstance(i, datetime.datetime) for i in self.index):
            raise TypeError("Resampling requires datetime.datetime index")
        return Resampler(self, rule)

    def rolling(self, window, min_periods=None):
        return Jandas.dataframe.Rolling(self, window, min_periods)

    def map(self, func):
        return Series([func(value) for value in self.data], self.index)


class Resampler:
    def __init__(self, series, rule):
        self.series = series
        self.rule = rule.upper()

    def _group_by_rule(self):
        grouped = defaultdict(list)

        for idx, val in zip(self.series.index, self.series.data):
            if self.rule == 'D':
                key = idx.date()
            elif self.rule == 'M':
                key = datetime.date(idx.year, idx.month, 1)
            elif self.rule == 'Y':
                key = datetime.date(idx.year, 1, 1)
            else:
                raise ValueError("Unsupported resample frequency: {}".format(self.rule))
            grouped[key].append(val)

        return grouped

    def mean(self):
        grouped = self._group_by_rule()
        means = []
        index = []

        for key in sorted(grouped.keys()):
            values = [v for v in grouped[key] if v is not None]
            if values:
                means.append(sum(values) / len(values))
                index.append(key)
            else:
                means.append(None)
                index.append(key)

        return Series(means, index=index, name=self.series.name)

    def sum(self):
        grouped = self._group_by_rule()
        totals = []
        index = []

        for key in sorted(grouped.keys()):
            values = [v for v in grouped[key] if v is not None]
            totals.append(sum(values) if values else 0)
            index.append(key)

        return Series(totals, index=index, name=self.series.name)
