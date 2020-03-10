from copy import deepcopy
import fastai.tabular as fast_tabular
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline


class CryptoDataset:
    """Face Landmarks dataset."""
    CLOSING_PRICE = 'close'
    PREV_CLOSING_PRICE = 'prev_close'
    DATE_COLUMN_NAME = 'date'
    ELAPSED = 'Elapsed'
    DROP_COLUMNS = ['Second', 'Minute', 'Month', 'Year',
                    'Is_quarter_end', 'Is_quarter_start',
                    'Is_year_end', 'Is_year_start',
                    'Is_month_end', 'Is_month_start',
                    'Week', 'Dayofyear', ELAPSED]
    TARGET_CLOSE_ = 'target_close_future_'

    def __init__(self, csv_file="BTC-ETH-filtered_with_indicators.csv", predict_delta=1, sequence_size=10,
                 batch_size=8):
        """
        Args:
            csv_file (string): Path to the csv file with the crypto stats table.
        """
        from datetime import datetime
        self.sequence_size = sequence_size
        self.df = pd.read_csv("BTC-ETH-filtered_with_indicators.csv",
                              # read dates as dates
                              parse_dates=[self.DATE_COLUMN_NAME],
                              date_parser=lambda x: datetime.fromtimestamp(int(x)),
                              )
        ### SETUP FOR GIVEN INPUT DATA
        # add the dateparts
        fast_tabular.add_datepart(self.df, self.DATE_COLUMN_NAME, time=True)
        # drop useless dateparts
        self.df = self.df.drop(columns=self.DROP_COLUMNS)

        # building the target
        self.df[[self.TARGET_CLOSE_]] = np.log(self.df[[self.CLOSING_PRICE]].shift(-predict_delta)) - np.log(
            self.df[[self.CLOSING_PRICE]])
        # remove last element since it is only used for creating the last prediction.
        self.df = self.df[:-predict_delta]
        self.__append_methods_to_dataframe(targets_names_list=[self.TARGET_CLOSE_])

        # train_test_val split of ds. self.train_df, self.valid_df, self.test_df will contain the splitted data
        self.train_df, self.valid_df, self.test_df = [], [], []
        self.__train_test_val_split(sequence_size=sequence_size)

        # cleanup the data, normalize using train set, clip outliers
        self.preprocessing_pipeline, self.labeling_pipeline = None, None  # normalization pipeline(s)
        self.train_data, self.test_data, self.valid_data = [None for _ in range(3)]
        self.train_target, self.valid_target, self.test_target = [None for _ in range(3)]
        self.__setup_normalize_data()

        # random loading of train input, but constant loading of validation and test inputs for reproducible tests
        self.train_loader, self.val_loader, self.test_loader = [None for _ in range(3)]
        self.setup_data_loaders(sequence_size=sequence_size, batch_size=batch_size)

    # Public

    def normalize_data(self, data_to_normalize_to_train_data: pd.DataFrame):
        data_to_normalize_to_train_data = deepcopy(data_to_normalize_to_train_data)
        
        # add the dateparts
        fast_tabular.add_datepart(data_to_normalize_to_train_data, self.DATE_COLUMN_NAME, time=True)
        # drop useless dateparts
        data_to_normalize_to_train_data.drop(columns=self.DROP_COLUMNS, inplace=True)
        
        # cleanup the data, normalize using train set, clip outliers
        columns = self.preprocessing_pipeline["poli-feature"].get_feature_names(self.train_df.input_data().columns)
        data_to_normalize_to_train_data = self.__transform_df(data_to_normalize_to_train_data.input_data(),
                                                              self.preprocessing_pipeline, columns=columns)
        
        data_to_normalize_to_train_data.drop(columns=['1'], inplace=True)
        return data_to_normalize_to_train_data

    # Private

    def __train_test_val_split(self, sequence_size=10):
        trvate_split = tuple(int(x * len(self.df)) for x in (0.75, 0.9, 1.0))
        tmp_train_limit = trvate_split[0] - trvate_split[0] % sequence_size
        self.train_df = self.df[:tmp_train_limit]
        tmp_valid_limit = trvate_split[1] - trvate_split[0] % sequence_size - (
                trvate_split[1] - trvate_split[0]) % sequence_size
        self.valid_df = self.df[tmp_train_limit:tmp_valid_limit]
        self.test_df = self.df[tmp_valid_limit:(len(self.df) - len(self.df) % sequence_size)]
        del tmp_train_limit, tmp_valid_limit

    def __append_methods_to_dataframe(self, targets_names_list=['next_closing_price']):
        def input_data(self: pd.DataFrame):
            return self.drop(columns=self.columns.intersection(['fake', 'volume_em', *targets_names_list]))

        pd.DataFrame.input_data = input_data

        def target(self: pd.DataFrame):
            return self[targets_names_list]

        pd.DataFrame.target = target

    def __setup_normalize_data(self):
        self.preprocessing_pipeline = Pipeline([("poli-feature", PolynomialFeatures(degree=2)),
                                           ("normalizer", StandardScaler())
                                           ]).fit(self.train_df.input_data(), self.train_df.target())
        columns = self.preprocessing_pipeline["poli-feature"].get_feature_names(self.train_df.input_data().columns)
        self.train_data = self.__transform_df(self.train_df.input_data(), self.preprocessing_pipeline, columns=columns)
        self.valid_data = self.__transform_df(self.valid_df.input_data(), self.preprocessing_pipeline, columns=columns)
        self.test_data = self.__transform_df(self.test_df.input_data(), self.preprocessing_pipeline, columns=columns)
        self.train_data.drop(columns=['1'], inplace=True)
        self.valid_data.drop(columns=['1'], inplace=True)
        self.test_data.drop(columns=['1'], inplace=True)

        self.labeling_pipeline = StandardScaler().fit(self.train_df.target())
        label_columns = self.train_df.target().columns
        self.train_target = self.__transform_df(self.train_df.target(), self.labeling_pipeline, columns=label_columns)
        self.valid_target = self.__transform_df(self.valid_df.target(), self.labeling_pipeline, columns=label_columns)
        self.test_target = self.__transform_df(self.test_df.target(), self.labeling_pipeline, columns=label_columns)
        # clip targetLabels to 90% (quantile)
        df = self.train_target.iloc[:, 0]
        no_outliers_mask = df.between(df.quantile(.05), df.quantile(.95))
        minimum_not_out = np.min(df[no_outliers_mask])
        maximum_not_out = np.max(df[no_outliers_mask])
        self.train_target = self.train_target.clip(lower=minimum_not_out, upper=maximum_not_out)
        self.valid_target = self.valid_target.clip(lower=minimum_not_out, upper=maximum_not_out)
        self.test_target = self.test_target.clip(lower=minimum_not_out, upper=maximum_not_out)

    def setup_data_loaders(self, sequence_size, batch_size):
        def create_inout_sequences(input_data: pd.DataFrame, labels: pd.DataFrame, seq_len: int):
            out_labels = torch.tensor(labels.iloc[seq_len - 1:].values.astype(np.float32))
            # print(input_data.shape)
            out_sequences = np.stack([input_data.iloc[i:i + seq_len] for i in range(len(input_data) - seq_len + 1)])
            out_data = torch.tensor(out_sequences.reshape(-1, seq_len, input_data.shape[-1]).astype(np.float32))
            # print(input_data.shape, out_data.shape, out_labels.shape)
            return out_data, out_labels

        #         train_x_y
        self.train_loader = DataLoader(dataset=TensorDataset(*create_inout_sequences(input_data=self.train_data,
                                                                                     labels=self.train_target,
                                                                                     seq_len=sequence_size)),
                                       batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=TensorDataset(*create_inout_sequences(input_data=self.valid_data,
                                                                                   labels=self.valid_target,
                                                                                   seq_len=sequence_size)),
                                     batch_size=batch_size)
        self.test_loader = DataLoader(dataset=TensorDataset(*create_inout_sequences(input_data=self.test_data,
                                                                                    labels=self.test_target,
                                                                                    seq_len=sequence_size)),
                                      batch_size=batch_size)

    # Static

    @staticmethod
    def __transform_df(df_to_transform, transformer, columns):
        return pd.DataFrame(transformer.transform(df_to_transform), columns=columns)
