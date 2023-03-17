import pathlib
import pandas as pd

from virny.custom_classes.base_dataset import BaseDataset


class CreditDataset(BaseDataset):
    def __init__(self, subsample_size=None):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        filename = 'givemesomecredit.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path, index_col=0)
        if subsample_size:
            df = df.sample(subsample_size)

        target = 'SeriousDlqin2yrs'
        numerical_columns = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                             'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                             'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines']
        categorical_columns = []
        features = numerical_columns + categorical_columns
        columns_with_nulls = df.columns[df.isna().any().to_list()].to_list()

        super().__init__(
            pandas_df=df,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            X_data=df[features],
            y_data=df[target],
            columns_with_nulls=columns_with_nulls,
        )
