class BaseDataset:
    def __init__(self, pandas_df, features, target, numerical_columns, categorical_columns,
                 X_data=None, y_data=None, columns_with_nulls=None):
        """

        :param pandas_df:
        :param features:
        :param target:
        :param numerical_columns:
        :param categorical_columns:
        :param X_data: optional
        :param y_data: optional
        :param columns_with_nulls: optional
        """
        self.dataset = pandas_df
        self.target = target
        self.features = features
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        self.X_data = self.dataset[features] if X_data is None else X_data
        self.y_data = self.dataset[target] if y_data is None else y_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list() \
            if columns_with_nulls is None else columns_with_nulls
