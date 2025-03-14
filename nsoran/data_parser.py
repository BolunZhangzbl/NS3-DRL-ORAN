# -- Public Imports
import os
import numpy as np
import pandas as pd

# -- Private Imports
from utils import *

# -- Global Variables

dict_kpms = dict(
    throughput='size',
    numusedrbs='...'
)

dict_columns = dict(
    tp=['time', 'cellId', 'IMSI', 'RNTI', 'mcs', 'size', 'rv', 'ndi', 'ccId'],      # Cell specific
    sinr=['time', 'cellId', 'IMSI', 'RNTI', 'rsrp', 'sinr', 'ComponentCarrierId']   # UE specific
)

dict_filenames = dict(
    tp='DlRxPhyStats.txt',
    sinr='DlRsrpSinrStats.txt'
)


# -- Functions

class DataParser:

    def __init__(self, args):

        self.time_step = args.time_step   # in ms
        self.num_enb = args.num_enb
        self.last_read_time = None        # Store the last processed timestamp

    def read_kpms(self, kpm_type='tp'):
        assert kpm_type in dict_columns.keys()
        columns = dict_columns.get(kpm_type)
        filename = dict_filenames.get(kpm_type)

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))

        try:
            df = pd.read_csv(file_path, delim_whitespace=True, comment='%', names=columns,
                             skiprows=1, index_col=False)
        except Exception as e:
            print(f"Error reading the file {filename}: {e}")
            return pd.DataFrame(), None

        # Sample rows where time == max(time)
        latest_time = df['time'].max()
        if self.last_read_time:
            df = df[df['time'] > self.last_read_time]

        if df.empty:
            return pd.DataFrame(), None

        # Update the last read timestamp
        self.last_read_time = latest_time

        # Process df
        df = self.drop_id_cols(df)
        df = self.filter_df_by_enbs(df)
        df = self.filter_df_by_kpms(df)
        df = self.fill_missing_cellid(df)

        df_results = df.sort_values(by=['cellId'])

        return df_results, latest_time

    def filter_df_by_enbs(self, df):
        """
        Filter df by CellId for each eNB only if we select multiple timestamps
        """
        df_results = df.groupby('cellId', as_index=False).sum()

        return df_results

    def filter_df_by_kpms(self, df):
        """
        Compute throughput for required KPMs
        """
        df_copy = df.copy()

        df_copy['throughput'] = df['size']/8

        return df_copy

    def drop_id_cols(self, df):
        """
        Drop ID columns
        """
        cols_to_drop = ['time', 'IMSI'] + [col for col in df.columns if 'Id' in col and col != 'cellId']
        df = df.drop(columns=cols_to_drop, errors='ignore')

        df_results = df.groupby('cellId', as_index=False).sum()

        return df_results

    def fill_missing_cellid(self, df):
        """
        Fill values for missing cellId with rows containing zeros for all columns.
        """
        # Ensure cellId is of integer type
        df['cellId'] = df['cellId'].astype(int)

        # Expected cell IDs
        expected_cell_ids = set(range(1, self.num_enb + 1))

        # Actual cell IDs in the DataFrame
        actual_cell_ids = set(df['cellId'].unique())

        # Check if all expected cell IDs are present
        if actual_cell_ids == expected_cell_ids:
            return df

        # Find missing cell IDs
        missing_cell_ids = expected_cell_ids - actual_cell_ids

        # Create a DataFrame for missing rows
        if missing_cell_ids:
            missing_data = {col: 0 for col in df.columns}  # Dictionary with all columns set to 0
            missing_rows = [{'cellId': cell_id, **missing_data} for cell_id in missing_cell_ids]
            missing_df = pd.DataFrame(missing_rows)

            # Concatenate the original DataFrame with the missing rows
            df = pd.concat([df, missing_df], ignore_index=True)

        # Sort the DataFrame by cellId for consistency
        df = df.sort_values(by='cellId').reset_index(drop=True)

        return df



# if __name__ == '__main__':
#     class Args:
#         time_step = 1  # 1 ms
#         num_enbs = 4  # 4 base stations (cell IDs 1 to 4)
#
#     args = Args()
#     parser = DataParser(args)
#
#     df_kpms = parser.read_kpms(filename="DlTxPhyStats.txt")
#     print(df_kpms)
#     print(np.array(df_kpms))
#
#     # Filter data by eNBs
#     df_filtered = parser.filter_df_by_enbs(df_kpms)
#     print(df_filtered)
#
#     # Compute throughput
#     df_with_tp = parser.filter_df_by_kpms(df_kpms)
#     print(df_with_tp)
