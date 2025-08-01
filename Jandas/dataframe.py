import csv
import io

from Jandas.series import Series
from Jandas.indexers import _LocIndexer, _IlocIndexer
from Jandas.vector import Vector


class GroupBy:
    """
    A class to represent grouped data in a DataFrame.

    Attributes:
    - groups: Dictionary where keys are group names and values are lists of rows in the group.
    - columns: List of column names for the grouped data.
    """

    def __init__(self, groups, columns, groupbyColumnIndices, numericOnly=False):
        """
        Initialize a GroupBy object.

        Parameters:
        - groups: Dictionary mapping group keys to lists of rows.
        - columns: List of column names for the grouped data.
        """
        self.groups = groups  # The grouped data
        self.columns = columns  # The column names of the DataFrame
        self.groupbyColumnIndices = groupbyColumnIndices
        self.groupIndex = JIndex(list(groups.keys()))  # NEW
        self.numeric_only = numericOnly

    def __getitem__(self, colName):
        if colName not in self.columns:
            raise KeyError("Column {} not found".format(colName))

        colIndex = self.columns.index(colName)

        # Build new groups with only the selected column values per row
        newGroups = {}
        for key, rows in self.groups.items():
            # extract just the values of colIndex for each row
            newGroups[key] = [[row[colIndex]] for row in rows]

        # The new GroupBy only has one column: colName
        return GroupBy(newGroups, [colName], groupbyColumnIndices=[])

    def mean(self, numeric_only=False):
        """
        Calculate the mean of each column in each group.

        Parameters:
        - numericOnly (bool): If True, only include numeric columns.

        Returns:
        - DataFrame with group keys as index and column means as values.
        """
        sampleRow = next(iter(self.groups.values()))[0]
        valueIndices = [
            i for i in range(len(self.columns))
            if i not in self.groupbyColumnIndices and (
                    not numeric_only or isinstance(sampleRow[i], (int, float))
            )
        ]
        valueCols = [self.columns[i] for i in valueIndices]

        data = []
        for key, group in self.groups.items():
            row = []
            for i in valueIndices:
                try:
                    values = [float(row[i]) for row in group]
                    meanVal = sum(values) / len(values)
                except (ValueError, TypeError):
                    meanVal = None
                row.append(meanVal)
            data.append(row)

        return DataFrame(data, columns=valueCols, index=JIndex(list(self.groups.keys())))

    def sum(self):
        """
        Calculate the sum of each column in each group.

        Returns:
        - Dictionary with group keys as keys and lists of column sums as values.
        """
        data = []
        for key, group in self.groups.items():
            row = [
                sum(float(x) for x in col)
                for i, col in enumerate(zip(*group))
                if i not in self.groupbyColumnIndices
            ]
            data.append(row)
        valueCols = [
            col for i, col in enumerate(self.columns)
            if i not in self.groupbyColumnIndices
        ]

        return DataFrame(data, columns=valueCols, index=JIndex(list(self.groups.keys())))

    def count(self):
        """
        Count the number of elements in each column for each group.

        Returns:
        - Dictionary with group keys as keys and lists of counts as values.
        """
        data = []
        for key, group in self.groups.items():
            row = [
                len(col)
                for i, col in enumerate(zip(*group))
                if i not in self.groupbyColumnIndices
            ]
            data.append(row)
        valueCols = [
            col for i, col in enumerate(self.columns)
            if i not in self.groupbyColumnIndices
        ]

        return DataFrame(data, columns=valueCols, index=JIndex(list(self.groups.keys())))

    def apply(self, func):
        """
        Apply a custom function to each column in each group.

        Parameters:
        - func: Function to apply to each column.

        Returns:
        - Dictionary with group keys as keys and lists of results as values.
        """
        data = []
        for key, group in self.groups.items():
            row = [
                func(col)
                for i, col in enumerate(zip(*group))
                if i not in self.groupbyColumnIndices
            ]
            data.append(row)
        valueCols = [
            col for i, col in enumerate(self.columns)
            if i not in self.groupbyColumnIndices
        ]

        return DataFrame(data, columns=valueCols, index=JIndex(list(self.groups.keys())))

    def getGroupKeys(self):
        """
        Get the keys of all groups.

        Returns:
        - List of group keys.
        """
        return list(self.groups.keys())


class DataFrame:
    """
    A simplified representation of a tabular data structure, similar to a spreadsheet or SQL table.

    This class stores data in rows, each encapsulated in a Vector, and maintains an ordered set of column names.
    It optionally supports a custom row index.

    Attributes:
    ----------
    data : List[Vector]
        The rows of the DataFrame, where each row is stored as a Vector with named fields.

    columns : JIndex
        The column labels for the DataFrame. Used for aligning and accessing column values.

    Index : Optional[List[Any]]
        An optional list representing custom row labels or indices.

    Initialization Notes:
    ---------------------
    - Data can be passed as a list of lists, a list of Vectors, or a dictionary mapping column names to lists.
    - If data is not provided, an empty table is created with the specified columns and index.
    - If data is a dictionary, each key is treated as a column and its associated values as column data.
    - Columns are automatically padded with None if they have unequal lengths.

    Raises:
    -------
    TypeError:
        If the input data format is unsupported.
    """

    def __init__(self, data=None, columns=None, index=None):
        """
        Initialize a DataFrame.

        Parameters:
        - data: 2D list or list of Vectors representing rows of data.
        - columns: List of column names.
        """
        self.Index = index

        if columns is None:
            columns = []
        if data is None:
            # Defer assigning data until we have both index and columns
            self.columns = JIndex(columns)
            numRows = len(index) if index is not None else 0
            self.data = [Vector([None] * len(columns), columns) for _ in range(numRows)]
            return
        if isinstance(data, dict):
            # Dict of columns: convert to list of rows
            columns = list(data.keys()) if not columns else columns
            colData = [data.get(col, []) for col in columns]
            numRows = max(len(col) for col in colData)
            # pad shorter columns with None
            paddedCols = [col + [None] * (numRows - len(col)) for col in colData]
            rows = zip(*paddedCols)
            self.data = [Vector(list(row), columns) for row in rows]
            self.columns = JIndex(columns)
            return
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], Vector):
                self.data = data
            else:
                self.data = [Vector(row, columns) for row in data]
            self.columns = JIndex(columns)
            return

        else:
            raise TypeError("Unsupported data type for DataFrame initialization.")

    @property
    def index(self):
        """
        Returns the index of the DataFrame.

        If a custom index (Index) has been set, it is returned.
        Otherwise, a default RangeIndex is returned based on the length of the data.
        """
        return self.Index if self.Index is not None else range(len(self.data))

    @index.setter
    def index(self, newIndex):
        """
        Set a new custom index for the DataFrame.

        Parameters:
        - newIndex: List of new index values. Must match the number of rows.
        """
        if len(newIndex) != len(self.data):
            raise ValueError("Length of new index must match number of rows in the DataFrame.")
        self.Index = newIndex

    @property
    def ndim(self):
        """
        Get the number of dimensions of the DataFrame.

        Returns:
        - 2 (always for a DataFrame).
        """
        return 2  # Since it's a DataFrame, it has 2 dimensions

    @property
    def size(self):
        """
        Get the total number of elements in the DataFrame.

        Returns:
        - Integer representing the total number of elements.
        """
        return len(self.data) * len(self.columns)

    @property
    def loc(self):
        """
        Get a _LocIndexer for label-based indexing.

        Returns:
        - _LocIndexer object.
        """
        return _LocIndexer(self, vectorCls=Vector, dataframeCls=DataFrame)

    @property
    def iloc(self):
        """
        Get an _IlocIndexer for position-based indexing.

        Returns:
        - _IlocIndexer object.
        """

        return _IlocIndexer(self)

    @property
    def axes(self):
        """
        Get the axes of the DataFrame (index and columns).

        Returns:
        - Named tuple containing index and columns as Series objects.
        """
        from collections import namedtuple
        Axis = namedtuple('Axis', ['index', 'columns'])
        index = Series(list(range(len(self.data))), name='index')
        columns = Series(self.columns, name='columns')
        return Axis(index=index, columns=columns)

    @property
    def values(self):
        """
        Get the values of the DataFrame as a 2D list.

        Returns:
        - 2D list of values.
        """
        # Return the 2D list of values (rows of vectors)
        return [row.as_array() for row in self.data]

    @property
    def shape(self):
        """
        Get the shape of the DataFrame.

        Returns:
        - Tuple (number of rows, number of columns).
        """
        return len(self.data), len(self.columns)

    @property
    def dtypes(self):
        """
        Get the data types of each column in the DataFrame.

        Returns:
        - List of data types for each column.
        """
        types = []
        for col in range(len(self.columns)):
            columnData = [row[col] for row in self.data]
            if all(isinstance(val, (int, float)) for val in columnData):
                types.append('float' if any(isinstance(val, float) for val in columnData) else 'int')
            else:
                types.append('object')

        # Create a Series to hold dtype info, with column names as the index
        dtypeSeries = Series(types, name='dtype')
        dtypeSeries.index = self.columns  # Set the column names as the index
        return dtypeSeries

    def rowByLabel(self, label):
        # Handle single-element tuple indexing
        if self.index and isinstance(self.index[0], tuple) and len(self.index[0]) == 1:
            label = (label,)
        idx = self.index.index(label)
        return self.data[idx]

    def set_index(self, columnName, inplace=False):
        """
        Set a specified column as the index of the DataFrame.

        Args:
            columnName (str): Name of the column to set as the index.
            inplace (bool): Whether to modify the DataFrame in place. If False, returns a new DataFrame.

        Returns:
            self: If inplace is False, returns a new DataFrame with the index set.

        Raises:
            ValueError: If the specified column name does not exist.
        """
        if columnName not in self.columns:
            raise ValueError("Column '{}' does not exist.".format(columnName))
        colIndex = self.columns.index(columnName)

        # Extract the new index values
        self.index = [row[colIndex] for row in self.data]

        # Create new rows without the index column
        self.data = [Vector([value for i, value in enumerate(row) if i != colIndex]) for row in self.data]

        # Update the column names
        self.columns = JIndex([col for i, col in enumerate(self.columns) if i != colIndex])
        if inplace:
            return
        else:
            return self

    def reset_index(self, name="index"):
        """
        Move the index into the first column of the DataFrame.

        Args:
            name (str): Column name for the index column.

        Returns:
            self: Modified DataFrame with index as the first column.
        """
        if self.index is None:
            raise ValueError("No index is set to reset.")

        # Insert the index values as the first column in each row
        self.data = [Vector([self.index[i]] + list(row)) for i, row in enumerate(self.data)]

        # Insert the column name for the index
        self.columns = [name] + self.columns

        # Clear the index
        self.index = range(len(self.data))
        return self

    def __len__(self):
        """
        Return the number of rows in the DataFrame.

        Returns:
            int: Number of rows in the DataFrame.
        """
        return len(self.data)

    def __getitem__(self, key):
        """
        Retrieve data from the DataFrame using various types of indexing.

        Args:
            key (str, int, tuple, slice, Series, DataFrame): Key to retrieve data.

        Returns:
            Series or DataFrame: Retrieved data based on the key.

        Raises:
            ValueError: If an invalid key or selector is provided.
            KeyError: If the key is unsupported.
        """
        if isinstance(key, str):
            # Return a column as a list
            colIndex = self.columns.index(key)
            return Series([row[colIndex] for row in self.data], index=self.index, name=key)
        elif isinstance(key, int):
            # Return a column by index as a list
            return Series([row[key] for row in self.data], self.columns[key])
        elif isinstance(key, tuple):
            # Multi-dimensional slicing
            rowSelector, colSelector = key

            # Select rows
            if isinstance(rowSelector, slice):
                selectedRows = self.data[rowSelector]
            elif isinstance(rowSelector, int):
                selectedRows = [self.data[rowSelector]]
            else:
                raise ValueError("Invalid row selector.")

            # Select columns
            if isinstance(colSelector, str):
                colIndex = self.columns.index(colSelector)
                return Series([row[colIndex] for row in selectedRows], colSelector)
            elif isinstance(colSelector, list):
                colIndices = [self.columns.index(col) for col in colSelector]
                selectedData = [[row[colIndex] for colIndex in colIndices] for row in selectedRows]
                return DataFrame(selectedData, [self.columns[i] for i in colIndices])
            else:
                raise ValueError("Invalid column selector.")
        elif isinstance(key, Series):
            # Boolean indexing
            filteredData = [row for row, include in zip(self.data, key.data) if include]
            return DataFrame(filteredData, self.columns)
        elif isinstance(key, list):
            # Boolean indexing
            filteredData = [row for row, include in zip(self.data, key.data) if include]
            return DataFrame(filteredData, self.columns)
        elif isinstance(key, slice):
            # Return rows as a new DataFrame
            return DataFrame(self.data[key], self.columns)
        elif isinstance(key, DataFrame):
            # Boolean indexing
            if key.data and all(isinstance(row[0], bool) for row in key.data):
                filteredData = [row for include, row in zip(key.data, self.data) if include[0]]
                return DataFrame(filteredData, self.columns)
            else:
                raise ValueError("Boolean indexing requires a DataFrame with boolean values.")
        else:
            raise KeyError("Unsupported key: {}".format(key))

    def __setitem__(self, key, value):
        """
        Set values in the DataFrame for a given column.

        Args:
            key (str): Column name to set values.
            value (iterable): Values to set in the column.

        Raises:
            KeyError: If the key is unsupported.
        """
        if isinstance(key, str):
            if key in self.columns:
                colIndex = self.columns.index(key)
                for rowIndex, val in enumerate(value):
                    self.data[rowIndex][colIndex] = val
            else:
                for rowIndex, val in enumerate(value):
                    self.data[rowIndex].to_list().append(val)
                oldCols = self.columns.to_list()
                oldCols.append(key)
                self.columns = JIndex(oldCols)
        else:
            raise KeyError("Unsupported key: {}".format(key))

    def __eq__(self, other):
        if not isinstance(other, DataFrame):
            return False
        return self.columns == other.columns and self.data == other.data

    def set_item_by_loc(self, key, value):
        """
        Set values in the DataFrame using .loc indexing.

        Args:
            key (tuple): Tuple of row and column selectors.
            value: Value(s) to set in the specified location.

        Raises:
            ValueError: If the column selector is invalid.
        """
        if isinstance(key, tuple):
            rowSelector, colSelector = key
            if isinstance(rowSelector, slice):
                rows = range(*rowSelector.indices(len(self.data)))
            else:
                rows = [rowSelector]

            if isinstance(colSelector, str):
                colIndex = self.columns.index(colSelector)
                for row in rows:
                    self.data[row][colIndex] = value
            elif isinstance(colSelector, list):
                colIndices = [self.columns.index(col) for col in colSelector]
                for row in rows:
                    for idx, colIndex in enumerate(colIndices):
                        self.data[row][colIndex] = value[idx]
            else:
                raise ValueError("Invalid column selector for .loc")
        elif isinstance(key, int):
            # Setting a whole row
            if len(value) != len(self.columns):
                raise ValueError("Length of value does not match number of columns.")
            if key < len(self.data):
                self.data[key] = list(value)
            elif key == len(self.data):
                self.data.append(list(value))
        else:
            raise ValueError("Invalid key for .loc")

    def set_item_by_iloc(self, key, value):
        """
        Set values in the DataFrame using .iloc indexing.

        Args:
            key (tuple): A tuple of row and column selectors.
                - Row selector can be an integer, slice, or list of integers.
                - Column selector can be an integer, slice, or list of integers.
            value: Value(s) to set in the specified locations.

        Raises:
            ValueError: If the column selector is invalid.
        """
        if isinstance(key, tuple):
            rowSelector, colSelector = key
            if isinstance(rowSelector, slice):
                rows = range(*rowSelector.indices(len(self.data)))
            else:
                rows = [rowSelector]

            if isinstance(colSelector, int):
                for row in rows:
                    self.data[row][colSelector] = value
            elif isinstance(colSelector, slice):
                colIndices = range(*colSelector.indices(len(self.columns)))
                for row in rows:
                    for colIndex in colIndices:
                        self.data[row][colIndex] = value
            elif isinstance(colSelector, list):
                for row in rows:
                    for idx, colIndex in enumerate(colSelector):
                        self.data[row][colIndex] = value[idx]
            else:
                raise ValueError("Invalid column selector for .iloc")

    def get_column(self, columnName):
        """
        Retrieve a column by name as a Vector.

        Args:
            columnName (str): The name of the column to retrieve.

        Returns:
            Vector: The data of the specified column.

        Raises:
            ValueError: If the column does not exist in the DataFrame.
        """
        try:
            # Find the index of the column name in the self.columns list
            index = self.columns.index(columnName)
        except ValueError:
            # If the column is not found, return None
            return None

        # Return the column name and the corresponding data
        return Vector([row[index] for row in self.data])

    def astype(self, dtypeOrDict):
        """
        Cast the DataFrame to the given data type(s).

        Args:
            dtypeOrDict (type or dict): A single type to cast all columns,
                                          or a dict mapping column names to types.

        Returns:
            JDataFrame: A new DataFrame with cast data.
        """
        newData = []
        colTypes = {}

        if isinstance(dtypeOrDict, dict):
            # Explicit mapping of column names to types
            for col in self.columns:
                colTypes[col] = dtypeOrDict.get(col, None)
        else:
            # One type for all columns
            for col in self.columns:
                colTypes[col] = dtypeOrDict

        for row in self.data:
            newRow = []
            for i, val in enumerate(row):
                col = self.columns[i]
                castType = colTypes.get(col)
                if castType:
                    try:
                        # Treat empty strings and 'NA' as None
                        if val in ("", "NA"):
                            newRow.append(None)
                        else:
                            newRow.append(castType(val))
                    except Exception as e:
                        raise ValueError(
                            "Could not cast column '{}' value '{}' to {}: {}".format(col, val, castType, e))
                else:
                    newRow.append(val)
            newData.append(Vector(newRow))

        return self.__class__(newData, columns=self.columns, index=self.index)

    def rolling(self, window, min_periods=None):
        """
           Return a Rolling object for performing rolling window operations.

           Parameters:
           - window (int): Size of the moving window. Must be >= 1.
           - min_periods (int, optional): Minimum number of non-null values in the window required to compute a result.
             Defaults to the window size.

           Returns:
           - Rolling: An object that supports aggregation methods like .mean(), .sum(), etc.,
             applied over rolling windows for each column.

           Example:
           >>> df.rolling(window=3, min_periods=2).mean()
           """
        return Rolling(self, window, min_periods)

    def to_csv(self, filepath=None):
        """
        Export the DataFrame to a CSV file or a string.

        Args:
            filepath (str or None): File path to save the CSV. If None, returns a CSV string.

        Returns:
            str: CSV data as a string if filepath is None.

        Raises:
            IOError: If there is an issue writing to the file.
        """
        if filepath is None:
            filepath = io.StringIO()  # Create an in-memory string buffer

        # Create a writer object (either to a file or to a StringIO object)
        if isinstance(filepath, str):
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write column headers
                writer.writerow(self.columns)
                # Write the data rows
                for row in self.data:
                    writer.writerow(row)
        else:
            writer = csv.writer(filepath)
            # Write column headers
            writer.writerow(self.columns)
            # Write the data rows
            for row in self.data:
                writer.writerow(row)

        # Return the CSV string if we used StringIO
        if isinstance(filepath, io.StringIO):
            return filepath.getvalue()

    def iterrows(self):
        """
        Iterate over the rows of the DataFrame.

        Yields:
            tuple: A tuple containing the row index and the row data as a Vector.
        """
        for i, row in enumerate(self.data):
            # Convert each row to a dictionary with column names as keys and values as Vectors
            rowVector = Vector([row[j] for j in range(len(row))], name="Row {}".format(i))
            yield i, rowVector

    def info(self):
        """
        Print a concise summary of the DataFrame similar to pandas.DataFrame.info().
        """
        buffer = ["<class 'Jandas.DataFrame'>",
                  "RangeIndex: {} entries, 0 to {}".format(len(self.data), len(self.data) - 1),
                  "Data columns (total {} columns):".format(len(self.columns))]

        for i, col in enumerate(self.columns):
            nonNullCount = sum(row[i] is not None for row in self.data)
            sampleValue = next((row[i] for row in self.data if row[i] is not None), None)
            dtype = type(sampleValue).__name__ if sampleValue is not None else 'NoneType'
            buffer.append(" {:<3} {:<20} {:>5} non-null   {}".format(i, col, nonNullCount, dtype))

        buffer.append("dtypes: {}".format(", ".join(sorted(set(
            type(row[i]).__name__ for row in self.data for i in range(len(self.columns)) if row[i] is not None
        )))))
        print("\n".join(buffer))

    def copy(self):
        """
        Create a deep copy of the DataFrame.

        Returns:
            DataFrame: A new DataFrame object with the same data and columns.
        """
        return DataFrame([row[:] for row in self.data], self.columns[:])

    def concat(self, other, axis=0):
        """
        Concatenate two DataFrames along a specified axis.

        Args:
            other (DataFrame): The DataFrame to concatenate.
            axis (int): The axis along which to concatenate (0 for rows, 1 for columns).

        Returns:
            DataFrame: The concatenated DataFrame.

        Raises:
            ValueError: If the axis is invalid or the DataFrames are incompatible.
        """
        if axis == 0:  # Concatenate vertically (add rows)
            return DataFrame(self.data + other.data, self.columns)
        elif axis == 1:  # Concatenate horizontally (add columns)
            if len(self.data) != len(other.data):
                raise ValueError("DataFrames must have the same number of rows to concatenate along axis 1")
            concatenatedData = [row1 + row2 for row1, row2 in zip(self.data, other.data)]
            return DataFrame(concatenatedData, self.columns + other.columns)
        else:
            raise ValueError("Invalid axis value. Axis must be 0 or 1.")

    def items(self):
        """
        Iterate over columns of the DataFrame as (columnName, Vector) pairs.

        Yields:
            tuple: Column name and column data as a Vector.
        """
        for colIndex, colName in enumerate(self.columns):
            colData = [row[colIndex] for row in self.data]
            yield colName, Vector(colData, name=colName)

    def keys(self):
        """
        Get the column names of the DataFrame.

        Returns:
            list: A list of column names.
        """
        return self.columns

    def diff(self, periods=1, axis=0):
        """
        Calculate the difference between rows or columns.

        Args:
            periods (int): Periods to shift for calculating difference.
            axis (int): 0 = row-wise diff (default), 1 = column-wise diff.

        Returns:
            DataFrame: A new DataFrame of differences.
        """
        if periods < 1:
            raise ValueError("periods must be >= 1")

        newData = []

        if axis == 0:
            # Row-wise diff
            for i in range(len(self.data)):
                if i < periods:
                    newData.append([None] * len(self.columns))
                else:
                    prevRow = self.data[i - periods]
                    currRow = self.data[i]
                    diffRow = []
                    for j in range(len(currRow)):
                        try:
                            diff = currRow[j] - prevRow[j]
                        except Exception:
                            diff = None
                        diffRow.append(diff)
                    newData.append(diffRow)

        elif axis == 1:
            # Column-wise diff
            for row in self.data:
                newRow = []
                for j in range(len(row)):
                    if j < periods:
                        newRow.append(None)
                    else:
                        try:
                            diff = row[j] - row[j - periods]
                        except Exception:
                            diff = None
                        newRow.append(diff)
                newData.append(newRow)

        else:
            raise ValueError("axis must be 0 (rows) or 1 (columns)")

        return DataFrame(newData, columns=self.columns)

    def pop(self, col):
        """
        Remove and return a column from the DataFrame.

        Args:
            col (str): The name of the column to remove.

        Returns:
            Vector: The data of the removed column.

        Raises:
            ValueError: If the column does not exist.
        """
        colIndex = self.columns.index(col)
        poppedData = [row.pop(colIndex) for row in self.data]
        self.columns.data.pop(colIndex)
        return Vector(poppedData)

    def tail(self, n=5):
        """
        Return the last `n` rows of the DataFrame.

        Parameters:
        n (int): Number of rows to return from the end of the DataFrame. Default is 5.

        Returns:
        DataFrame: A new DataFrame containing the last `n` rows.
        """
        return DataFrame(self.data[-n:], self.columns)

    def get(self, key, default=None):
        """
        Get the values of a specific column in the DataFrame.

        Parameters:
        key (str): The column name to retrieve.
        default: The value to return if the column does not exist. Default is None.

        Returns:
        Vector or default: A vector of values for the specified column or the default value if the column doesn't exist.
        """
        if key in self.columns:
            return Vector([row[self.columns.index(key)] for row in self.data])
        else:
            return default

    def isin(self, values):
        """
        Check if each element in the DataFrame is contained in the provided list of values.

        Parameters:
        values (iterable): The list or set of values to check for membership.

        Returns:
        DataFrame: A new DataFrame where each cell is a boolean indicating whether the value is in `values`.
        """
        result = []
        for row in self.data:
            result.append([cell in values for cell in row])
        return DataFrame(result, self.columns)

    def apply(self, func, axis=0):
        """
        Apply a function element-wise across a column or row.

        Parameters:
        func (function): The function to apply to the data.
        axis (int): Axis along which the function is applied. 0 for columns, 1 for rows.

        Returns:
        DataFrame: A new DataFrame with the function applied.

        Raises:
        ValueError: If the axis is not 0 or 1.
        """
        if axis == 0:  # Apply function to each column
            cols = list(zip(*self.data))  # Transpose rows to columns
            applied = [func(list(col)) for col in cols]
            return DataFrame([applied], columns=self.columns)
        elif axis == 1:  # Apply function to each row
            applied = [func(row) for row in self.data]
            return DataFrame([[val] for val in applied], columns=["result"], index=self.index)
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    def applymap(self, func):
        """
        Apply a function element-wise to all elements in the DataFrame.

        Parameters:
        func (function): The function to apply to each element.

        Returns:
        DataFrame: A new DataFrame with the function applied element-wise to each value.
        """
        return DataFrame([[func(value) for value in row.as_array()] for row in self.data], self.columns)

    def map(self, func):
        """
        Apply a function element-wise to each row in the DataFrame.

        Parameters:
        func (function): The function to apply to each row.

        Returns:
        DataFrame: A new DataFrame with the function applied to each element in each row.
        """
        return DataFrame([[func(value) for value in row] for row in self.data], self.columns)

    def groupby(self, columnNames, numericOnly=False):
        """
        Group the DataFrame by one or more columns.

        Parameters:
        columnNames (str | list): Column name or list of column names to group by.
        numericOnly (bool): Whether to include only numeric columns in aggregations.

        Returns:
        GroupBy: GroupBy object.
        """
        if isinstance(columnNames, str):
            columnNames = [columnNames]

        try:
            indices = [self.columns.index(col) for col in columnNames]
        except ValueError:
            return None

        groups = {}
        for row in self.data:
            key = tuple(row[i] for i in indices)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        return GroupBy(groups, self.columns, groupbyColumnIndices=indices, numericOnly=numericOnly)

    def corr(self):
        """
        Compute the Pearson correlation matrix of the DataFrame.

        Returns:
            DataFrame: Correlation matrix as a new DataFrame with columns and index as the original columns.
        """

        def mean(values):
            return sum(values) / len(values)

        def stddev(values, ddof=1):
            m = mean(values)
            variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
            return variance ** 0.5

        def pearsonCorr(x, y):
            meanX = mean(x)
            meanY = mean(y)
            numerator = sum((a - meanX) * (b - meanY) for a, b in zip(x, y))
            denominator = (sum((a - meanX) ** 2 for a in x) * sum((b - meanY) ** 2 for b in y)) ** 0.5
            if denominator == 0:
                return 0  # or None or float('nan'), depending on your choice
            return numerator / denominator

        cols = self.columns
        n = len(cols)
        # transpose data to get columns as lists
        columnsData = list(zip(*self.data))
        corrMatrix = []

        for i in range(n):
            rowCorr = []
            for j in range(n):
                corrValue = pearsonCorr(columnsData[i], columnsData[j])
                rowCorr.append(corrValue)
            corrMatrix.append(rowCorr)

        # Return as a DataFrame with columns and index
        return DataFrame(data=corrMatrix, columns=cols, index=cols)

    def count(self):
        """
        Count the number of non-null values in each column of the DataFrame.

        Returns:
        list: A list of counts for each column, where each count corresponds to the number of non-null values.
        """
        return Series([sum(1 for row in self.data if row[colIndex] is not None) for colIndex in
                       range(len(self.columns))], index=self.columns)

    def describe(self):
        """
        Generate descriptive statistics of the DataFrame.

        Returns:
            DataFrame: A DataFrame where each row is a statistic (mean, std, etc.)
                       and each column is a column in the original data.
        """
        stats = {
            "count": [],
            "mean": [],
            "std": [],
            "min": [],
            "25%": [],
            "50%": [],
            "75%": [],
            "max": []
        }

        # Calculate stats for each column
        for colIndex in range(len(self.columns)):
            columnData = [row[colIndex] for row in self.data if isinstance(row[colIndex], (int, float))]
            if columnData:
                stats["count"].append(len(columnData))
                stats["mean"].append(sum(columnData) / len(columnData))
                stats["std"].append(self._std(columnData))
                stats["min"].append(min(columnData))
                stats["25%"].append(self._percentile(columnData, 25))
                stats["50%"].append(self._percentile(columnData, 50))
                stats["75%"].append(self._percentile(columnData, 75))
                stats["max"].append(max(columnData))
            else:
                for key in stats:
                    stats[key].append(None)

        # Transpose to make stats the row index
        data = [[stats[stat][i] for stat in stats] for i in range(len(self.columns))]
        df = DataFrame(data, columns=list(stats.keys()))
        df.index = self.columns
        df_T = df.transpose()
        return df_T

    def max(self):
        """Return the maximum value for each column in the DataFrame."""
        return Series([max([row[colIndex] for row in self.data if row[colIndex] is not None]) for colIndex in
                       range(len(self.columns))], index=self.columns)

    def mean(self, axis=0):
        """Return the mean of the DataFrame's columns (axis=0) or rows (axis=1).

        Parameters:
        axis (int): Axis to calculate mean over. 0 for columns, 1 for rows.

        Returns:
        List: Mean values for columns or rows. Returns None for any row/column with no numeric values.

        Raises:
        ValueError: If axis is not 0 or 1.
        """

        def safeMean(values):
            numeric = [v for v in values if isinstance(v, (int, float))]
            return sum(numeric) / len(numeric) if numeric else None

        if axis == 0:
            return Series([safeMean(col) for col in zip(*self.values)], index=self.columns)
        elif axis == 1:
            return Series([safeMean(row) for row in self.data], index=self.columns)
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    def median(self):
        """Return the median value for each column in the DataFrame."""
        return Series([sorted([row[colIndex] for row in self.data if row[colIndex] is not None])[len(self.data) // 2]
                       for colIndex in range(len(self.columns))], index=self.columns)

    def min(self):
        """Return the minimum value for each column in the DataFrame."""
        return Series([min([row[colIndex] for row in self.data if row[colIndex] is not None]) for colIndex in
                       range(len(self.columns))], index=self.columns)

    def mode(self):
        """Return a new DataFrame with the mode(s) for each column."""
        from collections import Counter

        modeLists = []
        for colIndex in range(len(self.columns)):
            columnData = [row[colIndex] for row in self.data]
            counter = Counter(columnData)
            if not counter:
                modeLists.append([None])
                continue

            mostCommon = counter.most_common()
            maxCount = mostCommon[0][1]
            tiedValues = [value for value, count in mostCommon if count == maxCount]

            modeLists.append(tiedValues)

        # Find the longest mode list so we can pad shorter ones
        maxLen = max(len(modes) for modes in modeLists)

        # Transpose and pad with None
        resultData = []
        for i in range(maxLen):
            row = [modes[i] if i < len(modes) else None for modes in modeLists]
            resultData.append(row)

        return DataFrame(resultData, self.columns)

    def quantile(self, q, axis=0):
        """
        Return the q-th quantile value along the specified axis.

        Parameters:
        q (float): Quantile value to calculate (between 0 and 1).
        axis (int): Axis to compute quantile on.
                    0 = column-wise (default), 1 = row-wise

        Returns:
        List: Quantile values for each column (axis=0) or row (axis=1)
        """
        if not 0 <= q <= 1:
            raise ValueError("Quantile 'q' must be between 0 and 1.")

        def getQuantile(arr):
            arr = [v for v in arr if v is not None]
            if not arr:
                return None
            sortedArr = sorted(arr)
            index = int(len(sortedArr) * q)
            # Clamp to last index to avoid out-of-range
            index = min(index, len(sortedArr) - 1)
            return sortedArr[index]

        if axis == 0:
            # Compute quantile for each column
            return [
                getQuantile([row[i] for row in self.data])
                for i in range(len(self.columns))
            ]
        elif axis == 1:
            # Compute quantile for each row
            return [
                getQuantile(row)
                for row in self.data
            ]
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows)")

    def sum(self, axis=0):
        """Return the sum of the DataFrame's columns (axis=0) or rows (axis=1).

        Ignores non-numeric values (like strings or None).

        Parameters:
        axis (int): 0 for column-wise sum, 1 for row-wise sum.

        Returns:
        List[float]: Sum values for columns or rows.
        """
        if axis == 0:
            # Sum each column, ignoring non-numeric values
            return Series([
                sum(val for val in col if isinstance(val, (int, float)))
                for col in zip(*self.values)
            ], index=self.columns)
        elif axis == 1:
            # Sum each row, ignoring non-numeric values
            return Series([
                sum(val for val in row.as_array() if isinstance(val, (int, float)))
                for row in self.data
            ], index=self.index)
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")

    def __truediv__(self, other):
        """
        Support division of the DataFrame by a scalar (e.g., df / 1000).
        """
        if isinstance(other, (int, float)):
            newData = [row / other for row in self.data]
            return DataFrame(newData, columns=self.columns, index=self.index)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'DataFrame' and '{}'".format(type(other).__name__))

    def std(self):
        """Return the standard deviation for each column in the DataFrame.

        Returns:
            List[float or None]: Standard deviation for each column.
                                 Returns None for non-numeric columns or empty columns.
        """
        from math import sqrt

        stds = []
        for colIndex in range(len(self.columns)):
            # Extract the numeric values from this column
            values = [row[colIndex] for row in self.data if isinstance(row[colIndex], (int, float))]
            if values:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                stds.append(sqrt(variance))
            else:
                stds.append(None)
        return Series(stds, index=self.index)

    def _std(self, data):
        """Calculate the standard deviation for a given list of numeric values.

        Ignores non-numeric entries.

        Returns:
            float or None: Standard deviation of the numeric values, or None if none found.
        """
        values = [x for x in data if isinstance(x, (int, float))]
        if not values:
            return None
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5

    def _percentile(self, data, percentile):
        """Return the percentile value for a given data list."""
        data = sorted(data)
        index = ((len(data) - 1) * percentile / 100)
        if isinstance(index, int):
            return data[index]
        else:
            return (data[index.__floor__()] + data[index.__ceil__()]) / 2

    def nunique(self):
        """Return the number of unique values for each column in the DataFrame."""
        return [len(set(row[colIndex] for row in self.data if row[colIndex] is not None)) for colIndex in
                range(len(self.columns))]

    def value_counts(self, col):
        """Return the count of unique values for a specific column in the DataFrame.

        Parameters:
        col (str): The column name to count unique values.

        Returns:
        Counter: A dictionary-like object with the count of each unique value.
        """
        from collections import Counter
        colIndex = self.columns.index(col)
        return Counter(row[colIndex] for row in self.data if row[colIndex] is not None)

    def drop(self, labels, axis=0):
        """
        Drop specified rows or columns from the DataFrame.

        Parameters:
        labels (int or list): The row indices or column names to drop.
        axis (int): 0 for rows, 1 for columns.

        Returns:
        DataFrame: A new DataFrame with the specified rows or columns dropped.
        """
        if not isinstance(labels, list):
            labels = [labels]

        if axis == 0:
            # Drop rows by index
            newData = [row for i, row in enumerate(self.data) if self.Index[i] not in labels]
            newIndex = [self.Index[i] for i in range(len(self.data)) if self.Index[i] not in labels]

            return DataFrame(newData, self.columns, index=newIndex)

        elif axis == 1:
            # Drop columns by name
            colIndices = []
            for label in labels:
                matches = [i for i, col in enumerate(self.columns) if col == label]
                if not matches:
                    raise KeyError("Column '{}' not found.".format(label))
                colIndices.extend(matches)

            colIndices = sorted(set(colIndices), reverse=True)

            newData = []
            for row in self.data:
                newRow = [val for i, val in enumerate(row) if i not in colIndices]
                newData.append(newRow)

            newColumns = [col for i, col in enumerate(self.columns) if i not in colIndices]

            return DataFrame(newData, newColumns, index=self.Index)  # Preserve index

        else:
            raise ValueError("axis must be 0 (rows) or 1 (columns)")

    def dropna(self, axis=0, how='any', subset=None, inplace=False, ignoreIndex=False):
        """
        Drop rows or columns containing None values.

        Parameters:
            axis (int): 0 to drop rows, 1 to drop columns.
            how (str): 'any' (drop if any value is None), or 'all' (drop only if all are None).
            subset (list): List of column names to consider (only for axis=0).
            inplace (bool): Whether to modify the DataFrame in place. Default False.
            ignoreIndex (bool): Whether to reset the row indices (only affects axis=0). Default False.

        Returns:
            DataFrame or None: New DataFrame with rows or columns dropped, or None if inplace=True.

        Raises:
            ValueError: If axis is not 0 or 1, or if `how` is not 'any' or 'all'.
        """
        if how not in ('any', 'all'):
            raise ValueError("how must be 'any' or 'all'")

        if axis == 0:  # Drop rows
            if subset is not None:
                colIndices = [self.columns.index(col) for col in subset]
            else:
                colIndices = range(len(self.columns))

            if how == 'any':
                newData = [row for row in self.data if not any(row[i] is None for i in colIndices)]
            else:  # how == 'all'
                newData = [row for row in self.data if not all(row[i] is None for i in colIndices)]

            newColumns = list(self.columns)

            if ignoreIndex:
                # No row index handling needed unless you maintain an explicit index elsewhere
                pass

        elif axis == 1:  # Drop columns
            if subset is not None:
                raise ValueError("subset is only valid when axis=0")

            if how == 'any':
                validColumns = [i for i in range(len(self.columns)) if all(row[i] is not None for row in self.data)]
            else:  # how == 'all'
                validColumns = [i for i in range(len(self.columns)) if not all(row[i] is None for row in self.data)]

            newData = [[row[i] for i in validColumns] for row in self.data]
            newColumns = [self.columns[i] for i in validColumns]
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")

        if inplace:
            self.data = newData
            self.columns = newColumns
            return None
        else:
            return DataFrame(newData, newColumns)

    def fillna(self, value):
        """Fill all None values in the DataFrame with a specified value.

        Parameters:
        value (any): The value to replace None with.

        Returns:
        DataFrame: The DataFrame with None values replaced.
        """
        self.data = [[cell if cell is not None else value for cell in row] for row in self.data]
        return self

    def fill(self, value):
        """Alias for fillna."""
        return self.fillna(value)

    def isna(self):
        """Return a DataFrame with boolean values indicating None entries."""
        return DataFrame([Vector([cell is None for cell in row]) for row in self.data], self.columns)

    def isnull(self):
        """Alias for isna."""
        return self.isna()

    def notna(self):
        """Return a DataFrame with boolean values indicating non-None entries."""
        return DataFrame([[cell is not None for cell in row] for row in self.data], self.columns)

    def notnull(self):
        """Alias for notna."""
        return self.notna()

    def replace(self, to_replace, value):
        """
        Replace occurrences of a specified value or substring in the DataFrame with another value.

        Parameters:
        to_replace (any or str): The value or substring to replace.
        value (any or str): The value to replace with.

        Returns:
        DataFrame: The DataFrame with the replacements made.
        """

        def replaceCell(cell):
            if isinstance(cell, str) and isinstance(to_replace, str) and to_replace in cell:
                return cell.replace(to_replace, value)
            elif cell == to_replace:
                return value
            return cell

        self.data = [[replaceCell(cell) for cell in row] for row in self.data]
        return self

    def sort_values(self, by, ascending=True):
        """Sort the DataFrame by a specified column.

        Parameters:
        by (str): The column name to sort by.
        ascending (bool): Whether to sort in ascending order.

        Returns:
        DataFrame: The sorted DataFrame.
        """
        colIndex = self.columns.index(by)
        self.data.sort(key=lambda x: x[colIndex], reverse=not ascending)
        return self

    def sort_index(self, axis=0, ascending=True):
        """Sort the DataFrame by row or column index.

        Parameters:
        axis (int): Axis to sort by. 0 for rows, 1 for columns.
        ascending (bool): Whether to sort in ascending order.

        Returns:
        DataFrame: The sorted DataFrame.
        """
        # Sort rows by index
        if axis == 0:
            sortedPairs = sorted(zip(self.index, self.data), key=lambda pair: pair[0], reverse=not ascending)
            newIndex, newData = zip(*sortedPairs)
            return DataFrame(list(newData), columns=self.columns, index=JIndex(list(newIndex)))
        elif axis == 1:
            # Sort columns alphabetically
            sortedIndices = sorted(range(len(self.columns)), key=lambda i: self.columns[i], reverse=not ascending)
            newColumns = [self.columns[i] for i in sortedIndices]
            newData = [[row[i] for i in sortedIndices] for row in self.data]
            return DataFrame(newData, columns=newColumns, index=self.index)
        else:
            raise ValueError("axis must be 0 (rows) or 1 (columns)")

    def T(self):
        """Return the transpose of the DataFrame."""
        return self.transpose()

    def transpose(self):
        """Return the transpose of the DataFrame, similar to pandas.DataFrame.T."""

        # Ensure columns are strings
        self.columns = [str(col) for col in self.columns]

        # First, extract the values as a list of lists (convert any Row object to an array)
        values = [row.as_array() if hasattr(row, 'as_array') else row for row in self.data]

        # Transpose the data
        transposedData = list(map(list, zip(*values)))

        # Use original columns as new index, and row indices as new columns
        newColumns = self.index
        newIndex = self.columns

        # Create the transposed DataFrame
        df_t = DataFrame(transposedData, columns=newColumns)
        df_t.index = JIndex(newIndex)
        # if isinstance(df_t.columns, range):
        #     df_t.columns = list(df_t.columns)
        return df_t

    def join(self, other, on=None, how='left', lsuffix='', rsuffix=''):
        """
        Join the DataFrame with another DataFrame.

        Parameters:
        - other (DataFrame): The DataFrame to join with.
        - on (str): The column to join on. If None, join on the first column.
        - how (str): The type of join. Only 'left' join is currently supported.
        - lsuffix (str): Suffix to use for overlapping column names in the left DataFrame.
        - rsuffix (str): Suffix to use for overlapping column names in the right DataFrame.

        Returns:
        - DataFrame: The joined DataFrame.

        Raises:
        - NotImplementedError: If 'how' is not 'left'.
        """
        if how == 'left':
            # Handle column name collisions
            overlapping = set(self.columns) & set(other.columns)
            newSelfColumns = [col + lsuffix if col in overlapping else col for col in self.columns]
            newOtherColumns = [col + rsuffix if col in overlapping else col for col in other.columns]

            joinedData = []
            for row in self.data:
                if on:
                    joinValue = row[self.columns.index(on)]
                    matchingRow = next((r for r in other.data if r[other.columns.index(on)] == joinValue), [])
                    # If no matching row, fill with None for the other columns
                    if matchingRow is None:
                        matchingRow = [None] * len(other.columns)
                else:
                    matchingRow = other.data[0] if other.data else [None] * len(other.columns)
                joinedData.append(row + matchingRow)
            joinedColumns = newSelfColumns + newOtherColumns
            return DataFrame(joinedData, joinedColumns)
        else:
            raise NotImplementedError("Only 'left' join is currently implemented")

    def to_string(self):
        """
        Convert the DataFrame to a formatted string representation.

        This method converts the data in the DataFrame into a string format, aligning columns based on the maximum
        width of the values in each column. The column headers are included at the top, followed by the data rows.
        If the data or columns are empty, an empty string is returned.

        Returns:
            str: A string representation of the DataFrame.
        """
        # Determine the maximum width of each column for alignment
        if not self.data or not self.columns:
            return ""

        # Prepare a 2D list for the string representation
        rows = [[str(item) if item is not None else 'NaN' for item in row] for row in self.data]

        # Add column headers at the top
        headers = self.columns
        colWidths = [max(len(str(item)) for item in col) for col in zip(*([headers] + rows))]

        # Format rows with proper spacing
        formattedRows = [' '.join(header.ljust(colWidths[idx]) for idx, header in enumerate(headers))]

        for row in rows:
            formattedRows.append(' '.join(row[idx].ljust(colWidths[idx]) for idx in range(len(row))))

        return '\n'.join(formattedRows)

    def __str__(self):
        """
        String representation of the DataFrame.

        This method returns the string representation of the DataFrame by calling the `to_string` method.
        It is used when the `str()` function or print is called on an instance of the DataFrame class.

        Returns:
            str: A string representation of the DataFrame.
        """
        return self.to_string()

    def __repr__(self):
        """
        Official string representation of the DataFrame for debugging.

        This method provides a string representation that is suitable for debugging, showing the column names
        followed by the data in tab-separated format. This method is used when the `repr()` function is called
        or when an object is evaluated in an interactive Python session.

        Returns:
            str: A string representation of the DataFrame in a tabular format.
        """
        # Create the header row
        header = "\t" + "\t".join([str(x) for x in self.columns]) if self.columns else ""

        # Format the rows of data
        rows = "\n".join(
            "\t".join([str(self.index[i])] + [str(cell) for cell in self.data[i]])
            for i in range(len(self.data))
        )

        # Combine header and rows
        return "{}\n{}".format(header, rows) if header else rows

    def __add__(self, other):
        """
        Add another DataFrame or scalar to the DataFrame.

        If another DataFrame is provided, the corresponding values from both DataFrames are added element-wise. 
        If a scalar (int or float) is provided, the scalar is added to every element of the DataFrame.

        Args:
            other (DataFrame, int, float): The DataFrame or scalar to add.

        Returns:
            DataFrame: A new DataFrame with the result of the addition.

        Raises:
            ValueError: If the DataFrames have different shapes.
            TypeError: If the operand is not a DataFrame or scalar.
        """
        if isinstance(other, DataFrame):
            if len(self.columns) != len(other.columns) or len(self.data) != len(other.data):
                raise ValueError("DataFrames must have the same shape.")
            return DataFrame([[x + y for x, y in zip(row1.as_array(), row2.as_array())]
                              for row1, row2 in zip(self.data, other.data)], self.columns)
        elif isinstance(other, (int, float)):
            return DataFrame([[x + other for x in row.as_array()] for row in self.data], self.columns)
        else:
            raise TypeError("Unsupported operand type.")

    def __sub__(self, other):
        """
        Subtract another DataFrame or scalar from the DataFrame.

        If another DataFrame is provided, the corresponding values from both DataFrames are subtracted element-wise. 
        If a scalar (int or float) is provided, the scalar is subtracted from every element of the DataFrame.

        Args:
            other (DataFrame, int, float): The DataFrame or scalar to subtract.

        Returns:
            DataFrame: A new DataFrame with the result of the subtraction.

        Raises:
            ValueError: If the DataFrames have different shapes.
            TypeError: If the operand is not a DataFrame or scalar.
        """
        if isinstance(other, DataFrame):
            if len(self.columns) != len(other.columns) or len(self.data) != len(other.data):
                raise ValueError("DataFrames must have the same shape.")
            return DataFrame([[x - y for x, y in zip(row1.as_array(), row2.as_array())]
                              for row1, row2 in zip(self.data, other.data)], self.columns)
        elif isinstance(other, (int, float)):
            return DataFrame([[x - other for x in row.as_array()] for row in self.data], self.columns)
        else:
            raise TypeError("Unsupported operand type.")

    def __mul__(self, other):
        """
        Multiply the DataFrame by another DataFrame or scalar.

        If another DataFrame is provided, the corresponding values from both DataFrames are multiplied element-wise. 
        If a scalar (int or float) is provided, the scalar is multiplied by every element of the DataFrame.

        Args:
            other (DataFrame, int, float): The DataFrame or scalar to multiply.

        Returns:
            DataFrame: A new DataFrame with the result of the multiplication.

        Raises:
            ValueError: If the DataFrames have different shapes.
            TypeError: If the operand is not a DataFrame or scalar.
        """
        if isinstance(other, DataFrame):
            if len(self.columns) != len(other.columns) or len(self.data) != len(other.data):
                raise ValueError("DataFrames must have the same shape.")
            return DataFrame([[x * y for x, y in zip(row1.as_array(), row2.as_array())]
                              for row1, row2 in zip(self.data, other.data)], self.columns)
        elif isinstance(other, (int, float)):
            return DataFrame([[x * other for x in row.as_array()] for row in self.data], self.columns)
        else:
            raise TypeError("Unsupported operand type.")

    def head(self, n=5):
        """
        Return the first `n` rows of the DataFrame.

        This method returns a new DataFrame that contains the first `n` rows of the current DataFrame.

        Args:
            n (int): The number of rows to return. Defaults to 5.

        Returns:
            DataFrame: A new DataFrame containing the first `n` rows.
        """
        return DataFrame(self.data[:n], self.columns)


class JStringMethods(object):
    def __init__(self, index):
        self.index = index  # JIndex object
        self.data = index.data

    def _apply(self, func):
        return JIndex([func(val) for val in self.data])

    def contains(self, pat):
        return [pat in val for val in self.data]

    def startswith(self, prefix):
        return [val.startswith(prefix) for val in self.data]

    def endswith(self, suffix):
        return [val.endswith(suffix) for val in self.data]

    def upper(self):
        return self._apply(lambda x: x.upper())

    def lower(self):
        return self._apply(lambda x: x.lower())

    def replace(self, old, new):
        return self._apply(lambda x: x.replace(old, new))

    def strip(self):
        return self._apply(lambda x: x.strip())

    def match(self, regex):
        import re
        pattern = re.compile(regex)
        return [bool(pattern.match(val)) for val in self.data]


class JIndex(object):
    def __init__(self, data, name=None):
        self.data = list(data)
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return JIndex(self.data[key])
        elif isinstance(key, list):
            if all(isinstance(k, bool) for k in key):
                # Boolean mask
                return JIndex([v for v, m in zip(self.data, key) if m])
            else:
                return JIndex([self.data[k] for k in key])
        else:
            raise TypeError("Invalid index type for JIndex")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        content = ', '.join(repr(x) for x in self.data)
        return "JIndex([{}])".format(content)

    def __contains__(self, item):
        # Wrap item as tuple if index values are tuples of length 1
        if self.data and isinstance(self.data[0], tuple) and len(self.data[0]) == 1:
            item = (item,)
        return item in self.data

    def __add__(self, other):
        return JIndex(self.data + list(other))

    def __radd__(self, other):
        return JIndex(list(other) + self.data)

    def __eq__(self, other):
        if isinstance(other, JIndex):
            other = other.data
        return [a == b for a, b in zip(self.data, other)]

    def __getattr__(self, name):
        raise AttributeError(
            "'JIndex' object has no attribute '{}' (JIndex is immutable like pandas.Index)".format(name))

    def __setitem__(self, key, value):
        raise TypeError("JIndex does not support item assignment")

    # Optional: Add slicing support that returns JIndex
    def __getslice__(self, i, j):
        return JIndex(self._data[i:j])

    @property
    def str(self):
        return JStringMethods(self)

    @property
    def values(self):
        return self.data

    def to_list(self):
        return list(self.data)

    def index(self, val):
        return self.data.index(val)

    def append(self, value):
        raise AttributeError("JIndex has no attribute 'append' (immutable)")

    def pop(self, index=-1):

        raise AttributeError("JIndex has no attribute 'pop' (immutable)")

    def map(self, func):
        return JIndex([func(v) for v in self.data])

    def unique(self):
        seen = set()
        return JIndex([x for x in self.data if not (x in seen or seen.add(x))])

    def astype(self, dtype):
        if dtype == str:
            return JIndex([str(x) for x in self.data], name=self.name)
        else:
            raise NotImplementedError("astype currently only supports str")

    def is_unique(self):
        return len(set(self.data)) == len(self.data)

    def duplicated(self):
        seen = set()
        duplicates = set()
        return [x in seen and not (x in duplicates or duplicates.add(x)) or x in duplicates
                for x in self.data if not seen.add(x)]

    def nunique(self):
        return len(set(self.data))

    def intersection(self, other):
        return JIndex([x for x in self.data if x in other], name=self.name)

    def union(self, other):
        return JIndex(list(dict.fromkeys(self.data + list(other))), name=self.name)

    def difference(self, other):
        return JIndex([x for x in self.data if x not in other], name=self.name)

    def get_loc(self, val):
        if val not in self.data:
            raise KeyError("{} not found in JIndex".format(val))
        return [i for i, v in enumerate(self.data) if v == val]


class Rolling:
    def __init__(self, obj, window, min_periods=None):
        self.obj = obj
        self.window = window
        self.min_periods = window if min_periods is None else min_periods

    def mean(self):
        if isinstance(self.obj, DataFrame):
            return self._mean_dataframe()
        elif isinstance(self.obj, Series):
            return self._mean_series()
        else:
            raise TypeError("Rolling object must be a DataFrame or Series")

    def _mean_series(self):
        values = self.obj.data
        index = self.obj.index
        result = []

        # Clean and convert to float
        cleanValues = []
        for val in values:
            try:
                val = str(val).replace(',', '')
                cleanValues.append(float(val))
            except Exception:
                cleanValues.append(None)

        for i in range(len(cleanValues)):
            windowVals = cleanValues[max(0, i - self.window + 1):i + 1]
            validVals = [v for v in windowVals if v is not None]

            if len(validVals) >= self.min_periods:
                avg = sum(validVals) / len(validVals)
            else:
                avg = None

            result.append(avg)

        return Series(result, self.obj.name, index)

    def _mean_dataframe(self):
        resultRows = []
        numRows = len(self.obj.data)
        rollingData = {col: [] for col in self.obj.columns}

        for col in self.obj.columns:
            colValues = self.obj[col]
            cleanValues = []
            for val in colValues:
                try:
                    val = str(val).replace(',', '')
                    cleanValues.append(float(val))
                except Exception:
                    cleanValues.append(None)

            for i in range(numRows):
                windowVals = cleanValues[max(0, i - self.window + 1):i + 1]
                validVals = [v for v in windowVals if v is not None]

                if len(validVals) >= self.min_periods:
                    avg = sum(validVals) / len(validVals)
                else:
                    avg = None

                rollingData[col].append(avg)

        for i in range(numRows):
            rowValues = [rollingData[col][i] for col in self.obj.columns]
            resultRows.append(Vector(rowValues, self.obj.columns))

        return DataFrame(resultRows, list(self.obj.columns), self.obj.index)

    def __repr__(self):
        return "<Rolling window={}, min_periods={}>".format(self.window, self.min_periods)
