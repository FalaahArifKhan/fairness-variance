import pandas as pd
import numpy as np

from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime, ACSPublicCoverage, ACSMobility

from source.custom_classes.base_dataset import BaseDataset


class CompasDataset:
    def __init__(self, dataset_path):
        df = pd.read_csv(dataset_path)

        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        self.target = 'recidivism'
        self.numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        self.categorical_columns = ['race', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                                    'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        self.features = self.numerical_columns + self.categorical_columns

        self.X_data = df[self.features]
        self.y_data = df[self.target]
        self.dataset = df

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class CompasWithoutSensitiveAttrsDataset(BaseDataset):
    def __init__(self, dataset_path):
        # Read a dataset
        df = pd.read_csv(dataset_path)

        # Initial data types transformation
        int_columns = ['recidivism', 'age', 'age_cat_25 - 45', 'age_cat_Greater than 45',
                       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M', 'sex']
        int_columns_dct = {col: "int" for col in int_columns}
        df = df.astype(int_columns_dct)

        # Define params
        target = 'recidivism'
        numerical_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count']
        categorical_columns = ['age_cat_25 - 45', 'age_cat_Greater than 45','age_cat_Less than 25',
                                    'c_charge_degree_F', 'c_charge_degree_M']
        features = numerical_columns + categorical_columns

        super().__init__(
            pandas_df=df,
            features=features,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )


class ACSMobilityDataset:
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSMobility.features
        self.target = ACSMobility.target
        self.categorical_columns = ['MAR','SEX','DIS','ESP','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','COW','ESR']
        self.numerical_columns = ['AGEP', 'SCHL', 'PINCP', 'WKHP', 'JWMNP']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]

        self.y_data = acs_data[self.target].apply(lambda x: int(x == 1))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSPublicCoverageDataset:
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSPublicCoverage.features
        self.target = ACSPublicCoverage.target
        self.categorical_columns = ['MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','ESR','ST','FER','RAC1P']
        self.numerical_columns = ['AGEP', 'SCHL', 'PINCP']

        if with_nulls is True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
            
        self.y_data = acs_data[self.target].apply(lambda x: int(x == 1))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSTravelTimeDataset:
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSTravelTime.features
        self.target = ACSTravelTime.target
        self.categorical_columns = ['MAR','SEX','DIS','ESP','MIG','RELP','RAC1P','PUMA','ST','CIT','OCCP','POWPUMA','POVPIP']
        self.numerical_columns = ['AGEP', 'SCHL']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
            
        self.y_data = acs_data[self.target].apply(lambda x: int(x > 20))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()
    

class ACSIncomeDataset:
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSIncome.features
        self.target = ACSIncome.target
        self.categorical_columns = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
        self.numerical_columns = ['AGEP', 'SCHL']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
            
        self.y_data = acs_data[self.target].apply(lambda x: int(x > 50000))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSEmploymentDataset:
    def __init__(self, state, year, root_dir="data", with_nulls=False, optimize=True, subsample=None):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        if subsample !=None:
            acs_data = acs_data.sample(subsample)

        self.features = ACSEmployment.features
        self.target = ACSEmployment.target
        self.categorical_columns = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC','SCHL']
        self.numerical_columns = ['AGEP']

        if with_nulls is True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize==True:
            X_data = optimize_data_loading(X_data, self.categorical_columns)

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
        self.y_data = acs_data[self.target].apply(lambda x: int(x == 1))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSDataset_from_demodq:
    """ Following https://github.com/schelterlabs/demographic-data-quality """
    def __init__(self, state, year, with_nulls=False, optimize=True):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features =  ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        self.target = ['PINCP']
        self.categorical_columns = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
        self.numerical_columns = ['AGEP', 'WKHP']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        if optimize==True:
            X_data = optimize_data_loading(X_data, self.categorical_columns)

        self.X_data = X_data
        self.y_data = acs_data[self.target].apply(lambda x: x >= 50000).astype(int)

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


def optimize_data_loading(data, categorical):
    """
    Optimizing the dataset size by downcasting categorical columns
    """
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data
