import unittest

import Jandas
from Jandas.dataframe import DataFrame
from Jandas.series import Series
from Jandas.vector import Vector
import pandas as pd

class TestDataFrame(unittest.TestCase):

    def setUp(self):
        # Data as rows; each row corresponds to a Vector.
        self.df = DataFrame(
            data=[
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]
            ],
            columns=['A', 'B', 'C']
        )
        self.pd_df = pd.DataFrame(
            data=[
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9]
            ],
            columns=['A', 'B', 'C']
        )

    def test_apply(self):
        pd_result = self.pd_df.apply(sum)
        jandas_result = self.df.apply(sum)

        # Check each value matches
        for col in self.pd_df.columns:
            self.assertEqual(jandas_result[col], pd_result[col])

    def test_applymap(self):
        # Square every element
        pd_result = self.pd_df.map(lambda x: x * x)
        jandas_result = self.df.applymap(lambda x: x * x)

        # Check each element
        for row_idx in range(len(self.pd_df)):
            for col in self.pd_df.columns:
                self.assertEqual(jandas_result.iloc[row_idx][col], pd_result.iloc[row_idx][col])

    def test_astype(self):
        # Pandas
        expected = self.pd_df.astype(float)

        # Jandas
        result = self.df.astype(float)

        # Compare values
        for col in expected.columns:
            self.assertEqual(result[col], list(expected[col]))

    def test_axes(self):
        # Pandas
        expected_axes = self.pd_df.axes

        # Jandas
        result_axes = self.df.axes

        self.assertEqual(result_axes[0], list(range(len(self.df.data))))  # index
        self.assertEqual(result_axes[1], self.df.columns)  # columns
        self.assertEqual(result_axes[1], list(expected_axes[1]))

    def test_concat(self):
        # Pandas
        expected = pd.concat([self.pd_df, self.pd_df])

        # Jandas
        result = Jandas.concat([self.df, self.df])

        # Compare column values
        for col in expected.columns:
            self.assertEqual(list(result[col]), list(expected[col]))

    def test_copy(self):
        # Pandas
        expected = self.pd_df.copy()

        # Jandas
        result = self.df.copy()

        # Verify they have the same values
        for col in expected.columns:
            self.assertEqual(result[col], list(expected[col]))

        # Verify they are not the same object
        self.assertIsNot(result, self.df)

    def test_corr(self):
        # Pandas
        expected = self.pd_df.corr()

        # Jandas
        result = self.df.corr()

        # Compare correlation values
        for col in expected.columns:
            for row in expected.index:
                self.assertAlmostEqual(result[row][col], expected[row][col], places=5)

    def test_columns(self):
        self.assertEqual(self.df.columns, ['A', 'B', 'C'])
        self.assertEquals(list(self.pd_df.columns), ['A', 'B', 'C'])

    def test_count(self):
        # Pandas
        expected = self.pd_df.count()

        # Jandas
        result = self.df.count()

        # Compare column-wise non-null counts
        for col in expected.index:
            self.assertEqual(result[col], expected[col])

    def test_describe(self):
        # Pandas
        expected = self.pd_df.describe()

        # Jandas
        result = self.df.describe()

        # Compare basic statistics row by row
        for stat in expected.index:
            for col in expected.columns:
                self.assertAlmostEqual(result[col][stat], expected[col][stat], places=5)

    def test_diff(self):
        # Pandas
        expected = self.pd_df.diff()

        # Jandas
        result = self.df.diff()

        # Compare diff values element-wise
        for col in expected.columns:
            for i in range(len(expected[col])):
                val_pd = expected[col].iloc[i]
                val_jd = result[col][i]
                if pd.isna(val_pd):
                    self.assertIsNone(val_jd)  # or use np.nan depending on your implementation
                else:
                    self.assertAlmostEqual(val_jd, val_pd, places=5)

    def test_drop(self):
        new_df = self.df.drop(['B'], axis=1)
        pd_new_df = self.pd_df.drop(['B'], axis=1)
        self.assertEqual(new_df.columns, ['A', 'C'])
        self.assertEqual(list(pd_new_df.columns), ['A', 'C'])

    def test_dropna(self):
        # Add a row with NaN to both
        self.pd_df.loc[3] = [None, 10, 11]
        self.df.loc[3] = [None, 10, 11]

        # Pandas
        expected = self.pd_df.dropna()

        # Jandas
        result = self.df.dropna()

        # Compare values
        for col in expected.columns:
            self.assertEqual(result[col], list(expected[col]))

    def test_dtypes(self):
        # Pandas
        expected = self.pd_df.dtypes
        # expected = dict(self.pd_df.dtypes.apply(lambda dt: dt.name))

        # Jandas
        result = self.df.dtypes
        # result = dict(self.df.dtypes.apply(lambda dt: dt.name))

        # Compare dtypes as strings
        self.assertEqual(result, expected)

    def test_fill(self):
        # Add NaN row
        self.pd_df.loc[3] = [None, 10, 11]
        self.df.loc[3] = [None, 10, 11]

        # Fill NaN with 0
        expected = self.pd_df.fillna(0)
        result = self.df.fillna(0)

        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result.iloc[i][col], expected.iloc[i][col])

    def test_fillna(self):
        df = DataFrame({'A': [1, None, 3]})
        pd_df = pd.DataFrame({'A': [1, None, 3]})
        df_filled = df.fillna(0)
        pd_df_filled = pd_df.fillna(0)
        self.assertEqual(df_filled['A'], Series([1, 0, 3], name='A'))
        self.assertEqual(list(pd_df_filled['A'].values), list(pd.Series([1, 0, 3], name='A').values))

    def test_get(self):
        # Compare each row by index
        for i in range(len(self.pd_df)):
            expected = list(self.pd_df.iloc[:,i])
            result = list(self.df.get(self.pd_df.columns[i]))
            self.assertEqual(result, expected)

    def test_get_column(self):
        for col in self.pd_df.columns:
            expected = list(self.pd_df[col])
            result = list(self.df.get_column(col))
            self.assertEqual(result, expected)

    def test_groupby(self):
        # Add a categorical column to group by
        self.pd_df["group"] = ["X", "X", "Y"]
        self.df["group"] = ["X", "X", "Y"]

        # Group by and calculate mean
        expected = self.pd_df.groupby("group").mean(numeric_only=True)
        result = self.df.groupby("group").mean(numeric_only=True)

        for col in expected.columns:
            for group in expected.index:
                self.assertAlmostEqual(result.loc[group][col], expected.loc[group][col], places=5)

    def test_groupby_sum(self):
        df = DataFrame({
            'cat': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        pd_df = pd.DataFrame({
            'cat': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        grouped = df.groupby('cat').sum()
        pd_grouped = pd_df.groupby('cat').sum()
        self.assertEqual(grouped.loc['a'][0], 3)
        self.assertEqual(pd_grouped.loc['a'][0], 3)
        self.assertEqual(grouped.loc['b'][0], 7)
        self.assertEqual(pd_grouped.loc['b'][0], 7)

    def test_head(self):
        head = self.df.head(2)
        pd_head = self.pd_df.head(2)
        self.assertEqual(head.shape, (2, 3))
        self.assertEqual(pd_head.shape, (2, 3))

    def test_iloc(self):
        for i in range(len(self.pd_df)):
            expected = list(self.pd_df.iloc[i])
            result = list(self.df.iloc[i])
            self.assertEqual(result, expected)

    def test_iloc_single_row(self):
        row = self.df.iloc[1]
        pd_row = self.pd_df.iloc[1]
        assert isinstance(row, Series), "Expected a Series"
        assert row.index == ['A', 'B', 'C'], f"Unexpected index: {row.index}"
        assert list(row.data) == [2, 5, 8], f"Unexpected data: {row.data}"
        assert row.name == 1, f"Unexpected name: {row.name}"
        assert isinstance(pd_row, pd.Series), "Expected a Series"
        assert list(pd_row.index) == ['A', 'B', 'C'], f"Unexpected index: {pd_row.index}"
        assert list(pd_row.values) == [2, 5, 8], f"Unexpected data: {pd_row.data}"
        assert pd_row.name == 1, f"Unexpected name: {pd_row.name}"
        print("✅ test_iloc_single_row passed.")

    # def test_info(self):
    #     import io
    #     import sys
    #
    #     # Capture printed output
    #     captured_output = io.StringIO()
    #     sys.stdout = captured_output
    #     self.df.info()
    #     sys.stdout = sys.__stdout__
    #
    #     output = captured_output.getvalue()
    #
    #     # Check that output includes expected lines
    #     self.assertIn("Jandas DataFrame", output)
    #     self.assertIn("Number of columns:", output)
    #     self.assertIn("Number of rows:", output)
    #     for col in self.pd_df.columns:
    #         self.assertIn(col, output)

    def test_isin(self):
        values = [2, 4, 99]
        expected = self.pd_df.isin(values)
        result_df = self.df.isin(values)

        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result_df[col][i], expected.iloc[i][col])

    def test_isnull(self):
        # Add a row with None
        self.pd_df.loc[3] = [None, None, 10]
        self.df.loc[3] = [None, None, 10]

        expected = self.pd_df.isnull()
        result_df = self.df.isnull()

        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result_df[col][i], expected.iloc[i][col])

    def test_items(self):
        pd_items = dict(self.pd_df.items())
        jandas_items = dict(self.df.items())

        self.assertEqual(set(jandas_items.keys()), set(pd_items.keys()))

        for col in pd_items:
            self.assertEqual(list(jandas_items[col]), list(pd_items[col]))

    def test_iterrows(self):
        for (i1, row1), (i2, row2) in zip(self.df.iterrows(), self.pd_df.iterrows()):
            self.assertEqual(i1, i2)
            self.assertEqual(list(row1), list(row2))

    def test_join(self):
        # Create another DataFrame to join with
        other_data = [
            [100],
            [200],
            [300]
        ]
        other_columns = ['C']
        other_df = pd.DataFrame(other_data, columns=other_columns, index=self.pd_df.index)
        other = DataFrame(data=other_data, columns=other_columns, index=self.df.index)

        # Perform join
        expected = self.pd_df.join(other_df, rsuffix = 'new')
        result = self.df.join(other, rsuffix='new')

        for col in expected.columns:
            self.assertEqual(result[col], list(expected[col]))

    def test_keys(self):
        self.assertEqual(set(self.df.keys()), set(self.pd_df.keys()))

    def test_loc(self):
        self.df.index = ['x', 'y', 'z']
        self.pd_df.index = ['x', 'y', 'z']
        row = self.df.loc['y']
        pd_row = self.pd_df.loc['y']
        self.assertEqual(row, Vector([2, 5, 8],columns=['A','B','C']))
        self.assertEquals(list(pd_row.values), [2, 5, 8])

    def test_map(self):
        # Apply a simple map to column A (e.g., square each value)
        func = lambda x: x * x
        expected = self.pd_df['A'].map(func)
        result = self.df['A'].map(func)
        self.assertEqual(list(result), list(expected))

    def test_max(self):
        expected = self.pd_df.max()['A']
        result = self.df.max()['A']
        self.assertEqual(result, expected)

    def test_mean(self):
        result = self.df.mean()
        pd_result = self.pd_df.mean()
        self.assertEqual(result['A'], 2.0)
        self.assertEqual(pd_result['A'], 2.0)

    def test_median(self):
        expected = self.pd_df.median(numeric_only=True)['B']
        result = self.df.median()['B']
        self.assertEqual(result, expected)

    def test_min(self):
        expected = self.pd_df.min()['C']
        result = self.df.min()['C']
        self.assertEqual(result, expected)

    def test_mode(self):
        expected = self.pd_df.mode().iloc[0]['B']  # Use first row if multiple modes
        result = self.df.mode().iloc[0]['B']
        self.assertEqual(result, expected)

    def test_ndim(self):
        # Pandas DataFrame ndim
        pd_ndim = self.pd_df.ndim
        # Jandas DataFrame ndim
        jandas_ndim = self.df.ndim

        self.assertEqual(jandas_ndim, pd_ndim, "ndim property should match pandas")

    def test_notna(self):
        # Pandas DataFrame notna returns a DataFrame of booleans
        pd_notna = self.pd_df.notna()
        jandas_notna = self.df.notna()

        # Convert pandas notna to list of lists for comparison
        pd_notna_list = list(pd_notna.values)

        self.assertEqual(list(jandas_notna), pd_notna_list, "notna() output should match pandas")

    def test_notnull(self):
        # notnull is an alias of notna in pandas, so test similarly
        pd_notnull = self.pd_df.notnull()
        jandas_notnull = self.df.notnull()

        self.assertEqual(list(jandas_notnull), list(pd_notnull), "notnull() output should match pandas")

    def test_nunique(self):
        # Pandas nunique returns number of unique values per column by default
        pd_nunique = self.pd_df.nunique()
        jandas_nunique = self.df.nunique()

        # pd_nunique is a Series, convert to dict for comparison
        pd_nunique_dict = pd_nunique.to_dict()

        # Assuming jandas.nunique returns a dict or similar mapping column->count
        self.assertEqual(list(jandas_nunique), list(pd_nunique), "nunique() output should match pandas")

    def test_pop(self):
        # Pop a column from pandas df
        pd_pop = self.pd_df.pop('B')
        # Pop a column from jandas df
        jandas_pop = self.df.pop('B')

        # Check popped column values equal
        self.assertEqual(jandas_pop.data, pd_pop.tolist(), "pop() returned values should match pandas")
        # Check the column is removed from both dataframes
        self.assertNotIn('B', self.df.columns, "popped column should be removed in jandas")
        self.assertNotIn('B', self.pd_df.columns, "popped column should be removed in pandas")

    def test_quantile(self):
        # Pandas quantile for 0.5 (median)
        pd_quantile = self.pd_df.quantile(0.5)
        # Jandas quantile for 0.5
        jandas_quantile = self.df.quantile(0.5)

        # pandas returns a Series, convert to dict
        pd_quantile_dict = pd_quantile.to_dict()

        # jandas quantile expected to return dict or similar
        self.assertEqual(list(jandas_quantile), list(pd_quantile), "quantile() output should match pandas")

    def test_replace(self):
        # Replace 4 with 40 in pandas
        pd_replaced = self.pd_df.replace(4, 40)
        # Replace 4 with 40 in jandas
        jandas_replaced = self.df.replace(4, 40)

        # Check data equality (list of lists)
        self.assertEqual(jandas_replaced.data, pd_replaced.values.tolist(),
                         "replace() should modify values like pandas")

    def test_reset_index(self):
        # Reset index in pandas (default drop=False)
        pd_reset = self.pd_df.reset_index(drop=True)
        # Reset index in jandas
        jandas_reset = self.df.reset_index()

        # Convert pandas DataFrame to list of lists for comparison
        pd_reset_data = list(pd_reset.values)
        pd_reset_cols = list(pd_reset.columns)

        # Check columns match
        self.assertEqual(jandas_reset.columns, pd_reset.columns, "reset_index() columns should match pandas")
        # Check data matches
        for row_idx in range(len(self.pd_df)):
            for col in self.pd_df.columns:
                self.assertEqual(jandas_reset.iloc[row_idx][col], pd_reset.iloc[row_idx][col])

    def test_rolling(self):
        pd_rolled = self.pd_df.rolling(2).mean()
        df_rolled = self.df.rolling(2).mean()

        for i in range(len(self.df.data)):
            for j, col in enumerate(self.df.columns):
                pd_val = pd_rolled.iloc[i, j]
                jandas_val = df_rolled.data[i][j]
                if pd.isna(pd_val):
                    self.assertIsNone(jandas_val)
                else:
                    self.assertAlmostEqual(jandas_val, pd_val)

    def test_row_by_label(self):
        for i in range(len(self.df.data)):
            self.assertEqual(list(self.df.rowByLabel(i)), self.pd_df.iloc[i].tolist())

    def test_set_index(self):
        self.df.set_index('A', inplace=True)
        self.pd_df.set_index('A', inplace=True)

        self.assertEqual(self.df.columns, list(self.pd_df.columns))
        self.assertEqual(self.df.index, list(self.pd_df.index))

        for label in self.df.index:
            self.assertEqual(list(self.df.loc[label]), self.pd_df.loc[label].tolist())

    def test_shape(self):
        self.assertEqual(self.df.shape, (3, 3))
        self.assertEqual(self.pd_df.shape, (3, 3))

    def test_size(self):
        self.assertEqual(self.df.size, self.pd_df.size)

    def test_sort_index(self):
        expected = self.pd_df.sort_index()
        result = self.df.sort_index()
        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result[col][i], expected.iloc[i][col])


    def test_sort_values(self):
        expected = self.pd_df.sort_values(by='B', ascending=False)
        result = self.df.sort_values(by='B', ascending=False)
        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result[col][i], expected.iloc[i][col])

    def test_std(self):
        expected = self.pd_df.std()
        result = self.df.std()
        self.assertEqual(list(expected.values), result.values)

    def test_sum(self):
        result = self.df.sum()
        pd_result = self.pd_df.sum()
        self.assertEqual(result['A'], 6)
        self.assertEqual(result['B'], 15)
        self.assertEqual(result['C'], 24)
        self.assertEqual(pd_result['A'], 6)
        self.assertEqual(pd_result['B'], 15)
        self.assertEqual(pd_result['C'], 24)

    def test_tail(self):
        expected = self.pd_df.tail(2)
        result = self.df.tail(2)
        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result[col][i], expected.iloc[i][col])

    def test_transpose(self):
        expected = self.pd_df.T
        result = self.df.transpose()
        for col in expected.columns:
            for i in range(len(expected)):
                self.assertEqual(result[col][i], expected.iloc[i][col])
    # def test_iloc(self):
    #     row = self.df.iloc[1]
    #     self.assertEqual(row, self.pd_df.iloc[1])



import timeit
import unittest
import pandas as pd

class TestSpeedBenchmark(unittest.TestCase):
    def setUp(self):
        # Prepare data
        self.rows = 100
        self.cols = 10
        self.data = [[(i + j) for j in range(self.cols)] for i in range(self.rows)]
        self.columns = [f'col{j}' for j in range(self.cols)]

        # Jandas
        self.jandas_df = DataFrame(self.data, columns=self.columns)

        # Pandas
        self.pandas_df = pd.DataFrame(self.data, columns=self.columns)

    def test_speed_benchmark(self):
        print("\nSpeed benchmark (100 runs each):")

        # Pandas head
        pandas_head_time = timeit.timeit(lambda: self.pandas_df.head(5), number=100)
        print(f"Pandas head(): {pandas_head_time:.6f} sec")

        # Jandas head
        jandas_head_time = timeit.timeit(lambda: self.jandas_df.head(5), number=100)
        print(f"Jandas head(): {jandas_head_time:.6f} sec")

        # Pandas sort
        pandas_sort_time = timeit.timeit(lambda: self.pandas_df.sort_values(by='col5'), number=100)
        print(f"Pandas sort_values(): {pandas_sort_time:.6f} sec")

        # Jandas sort
        jandas_sort_time = timeit.timeit(lambda: self.jandas_df.sort_values(by='col5'), number=100)
        print(f"Jandas sort_values(): {jandas_sort_time:.6f} sec")

        # Pandas transpose
        pandas_transpose_time = timeit.timeit(lambda: self.pandas_df.transpose(), number=100)
        print(f"Pandas transpose(): {pandas_transpose_time:.6f} sec")

        # Jandas transpose
        jandas_transpose_time = timeit.timeit(lambda: self.jandas_df.transpose(), number=100)
        print(f"Jandas transpose(): {jandas_transpose_time:.6f} sec")

        # Pandas iloc
        pandas_iloc_time = timeit.timeit(lambda: self.pandas_df.iloc[2:7], number=100)
        print(f"Pandas iloc[]: {pandas_iloc_time:.6f} sec")

        # Jandas iloc
        jandas_iloc_time = timeit.timeit(lambda: self.jandas_df.iloc[2:7], number=100)
        print(f"Jandas iloc[]: {jandas_iloc_time:.6f} sec")

        # Pandas describe
        pandas_describe_time = timeit.timeit(lambda: self.pandas_df.describe(), number=100)
        print(f"Pandas describe(): {pandas_describe_time:.6f} sec")

        # Jandas describe
        jandas_describe_time = timeit.timeit(lambda: self.jandas_df.describe(), number=100)
        print(f"Jandas describe(): {jandas_describe_time:.6f} sec")