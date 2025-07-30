import Jandas
from Jandas.series import Series


class _LocIndexer:
    def __init__(self, dataframe, vectorCls=None, dataframeCls=None):
        self.dataframe = dataframe
        self.Vector = vectorCls
        self.DataFrame = dataframeCls

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rowSelector, colSelector = key
        else:
            rowSelector = key
            colSelector = slice(None)
        print(self.dataframe.index)
        print(rowSelector in self.dataframe.index)
        # --- ROW SELECTION ---
        if isinstance(rowSelector, slice):
            selectedRows = self.dataframe.data[rowSelector]
        elif rowSelector in self.dataframe.index:
            return self.dataframe.rowByLabel(key)
        elif isinstance(rowSelector, int):
            row = self.dataframe.data[rowSelector]

            # --- COLUMN SELECTION FOR SINGLE ROW ---
            colSelector = self._resolve_col_selector(colSelector)
            if isinstance(colSelector, list):
                return [row[idx] for idx in colSelector]
            elif isinstance(colSelector, int):
                return row[colSelector]

            else:
                raise ValueError("Invalid column selector for single row.")
        else:
            raise ValueError("Row selector must be an int or slice.")

        # --- COLUMN SELECTION FOR MULTIPLE ROWS ---
        colSelector = self._resolve_col_selector(colSelector)

        if isinstance(colSelector, list):
            dfNew = self.DataFrame([[row[idx] for idx in colSelector] for row in selectedRows],
                                   [self.dataframe.columns[idx] for idx in colSelector])
            dfNew._index = self.dataframe.index
            return dfNew
        elif isinstance(colSelector, int):
            return [row[colSelector] for row in selectedRows]
        else:
            return selectedRows  # All columns

    def __setitem__(self, key, value):
        self.dataframe.set_item_by_loc(key, value)

    def _resolve_col_selector(self, colSelector):
        # Normalize the column selector to a list of column indices
        columns = self.dataframe.columns

        if isinstance(colSelector, str):
            return columns.index(colSelector)
        elif isinstance(colSelector, list):
            # Handle boolean masks
            if all(isinstance(c, bool) for c in colSelector):
                if len(colSelector) != len(columns):
                    raise ValueError("Boolean mask length must match number of columns.")
                return [i for i, flag in enumerate(colSelector) if flag]
            else:
                return [columns.index(col) for col in colSelector]
        elif isinstance(colSelector, slice):
            return list(range(*colSelector.indices(len(columns))))
        elif colSelector == slice(None):
            return list(range(len(columns)))
        else:
            raise ValueError("Unsupported column selector")


class _IlocIndexer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rowSelector, colSelector = key
        else:
            # If it's a single index, assume it's for rows (or columns)
            rowSelector = key
            colSelector = slice(None)  # Assuming we're selecting all columns in this case

        # Handle single row
        if isinstance(rowSelector, int):
            row = self.dataframe.data[rowSelector]

            if isinstance(colSelector, int):
                return row[colSelector]

            elif isinstance(colSelector, slice):
                data = row[colSelector]
                cols = self.dataframe.columns[colSelector]
                return Series(data=data, index=cols, name=rowSelector)

            elif isinstance(colSelector, list):
                data = [row[i] for i in colSelector]
                cols = [self.dataframe.columns[i] for i in colSelector]
                return Series(data=data, index=cols, name=rowSelector)

            else:
                raise ValueError("Invalid column selector in .iloc[].")

        # Handle multiple rows
        elif isinstance(rowSelector, slice):
            selectedRows = self.dataframe.data[rowSelector]

            if isinstance(colSelector, int):
                return [row[colSelector] for row in selectedRows]

            elif isinstance(colSelector, slice):
                newData = [row[colSelector] for row in selectedRows]
                newColumns = self.dataframe.columns[colSelector]
                return Jandas.DataFrame(newData, columns=newColumns)

            elif isinstance(colSelector, list):
                return [[row[i] for i in colSelector] for row in selectedRows]

            else:
                raise ValueError("Invalid column selector in .iloc[].")

        else:
            raise ValueError("Row selector must be int or slice.")

    def __setitem__(self, key, value):
        self.dataframe.set_item_by_iloc(key, value)


class MultiIndex:
    def __init__(self, keys):
        # Normalize keys
        if not keys:
            self.keys = []
            self.nlevels = 0
        elif isinstance(keys[0], tuple):
            self.keys = keys
            self.nlevels = len(keys[0])
        else:
            # Single-level index, wrap each item in a tuple
            self.keys = [(k,) for k in keys]
            self.nlevels = 1

        self.levels = self._compute_levels()
        self.codes = self._compute_codes()

    def _compute_levels(self):
        levels = []
        for i in range(self.nlevels):
            level_i = sorted(set(k[i] for k in self.keys))
            levels.append(level_i)
        return levels

    def _compute_codes(self):
        codes = []
        for k in self.keys:
            codes.append([self.levels[i].index(k[i]) for i in range(self.nlevels)])
        return codes

    def __getitem__(self, idx):
        if self.nlevels == 1:
            # Return scalar for single-level index
            return self.keys[idx][0]
        return self.keys[idx]

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        header = "MultiIndex with {} entries and {} level{}\n".format(len(self), self.nlevels,
                                                                      's' if self.nlevels > 1 else '')
        preview = "\n".join(str(self[i]) for i in range(min(10, len(self))))
        return header + preview + ("\n..." if len(self.keys) > 10 else "")
