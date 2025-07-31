class Vector:
    def __init__(self, data, columns=None, name=None):
        """Initialize the Vector with data."""
        if isinstance(data, list):
            self.data = data
            self.columns = columns
            self.name = name
        else:
            raise TypeError("Data must be a list.")

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return self.data == other.data

    def __repr__(self):
        dtype = type(self.data[0]).__name__ if self.data else "float"
        index = range(len(self.data))
        formattedData = "\n".join("{}    {}".format(i, value) for i, value in zip(index, self.data))
        return "{}\nName: {}, dtype: {}".format(formattedData, self.name, dtype)

    def __len__(self):
        """Get the length of the vector."""
        return len(self.data)

    def __getitem__(self, index):
        """Indexing support."""
        if isinstance(index, str):  # If indexing by column name
            if self.columns is None:
                raise ValueError("Column names are not defined.")
            colIndex = self.columns.index(index)  # Find the column index
            return self.data[colIndex]
        elif isinstance(index, int):  # If indexing by position
            return self.data[index]
        elif isinstance(index, slice):
            return Vector(self.data[index])
        else:
            raise TypeError("Index must be an integer or a column name.")

    def __setitem__(self, index, value):
        """Support for setting elements by index."""
        if isinstance(index, str):  # Setting value by column name
            if self.columns is None:
                raise ValueError("Column names are not defined.")
            colIndex = self.columns.index(index)
            self.data[colIndex] = value
        elif isinstance(index, int):  # Setting value by position
            self.data[index] = value
        else:
            raise TypeError("Index must be an integer or a column name.")

    def __delitem__(self, index):
        del self.data[index]

    def _apply_operation(self, other, operation):
        """Helper to apply element-wise operation."""
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError("Vectors must be the same length.")
            return Vector([operation(x, y) for x, y in zip(self.data, other.data)])
        elif isinstance(other, (int, float)):
            return Vector([operation(x, other) for x in self.data])
        else:
            raise TypeError("Unsupported operand type.")

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data + other.data)  # Combine two vectors
        elif isinstance(other, list):
            return Vector(self.data + other)  # Combine vector with list
        else:
            raise TypeError("Unsupported operand type for +: 'Vector' and '{}'".format(type(other).__name__))

    def __sub__(self, other):
        return self._apply_operation(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._apply_operation(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._apply_operation(other, lambda x, y: x / y)

    def __floordiv__(self, other):
        return self._apply_operation(other, lambda x, y: x // y)

    def __mod__(self, other):
        return self._apply_operation(other, lambda x, y: x % y)

    def __pow__(self, other):
        return self._apply_operation(other, lambda x, y: x ** y)

    def sum(self):
        """Return the sum of elements in the vector."""
        return sum(self.data)

    def mean(self):
        """Return the mean of elements in the vector."""
        return sum(self.data) / len(self.data) if self.data else float('nan')

    def min(self):
        """Return the minimum value in the vector."""
        return min(self.data)

    def max(self):
        """Return the maximum value in the vector."""
        return max(self.data)

    def std(self):
        """Return the standard deviation of elements."""
        mean = self.mean()
        return (sum([(x - mean) ** 2 for x in self.data]) / len(self.data)) ** 0.5

    def dot(self, other):
        """Return the dot product with another vector."""
        if len(self) != len(other):
            raise ValueError("Vectors must be the same length for dot product.")
        return sum(x * y for x, y in zip(self.data, other.data))

    def transpose(self):
        """Return the transposed vector (just a copy in 1D)."""
        return self  # For 1D vector, transpose is trivial.

    def apply(self, func):
        """Apply a function element-wise."""
        return Vector([func(x) for x in self.data])

    def __iter__(self):
        """Iterator for the vector."""
        return iter(self.data)

    def as_array(self):
        """Return the raw list (or array in Java terms)."""
        return self.data

    def to_list(self):
        # Simply return the data attribute as a list
        return self.data

    def map(self, func):
        """Apply a function to each element of the vector and return a new vector."""
        return Vector([func(x) for x in self.data])

    def pop(self, index):
        """Remove and return the element at the specified index."""
        return self.data.pop(index)

    @staticmethod
    def from_array(arr):
        """Convert a raw array (list) to a Vector."""
        return Vector(arr)
