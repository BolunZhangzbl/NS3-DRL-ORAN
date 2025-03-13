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


# -- Functions

class DataParser:

    global last

    def __init__(self, args):

        self.time_step = args.time_step   # in ms
        self.num_enbs = args.num_enbs

        self.column_state = ['time', 'cellId', 'IMSI', 'RNTI', 'mcs', 'size', 'rv', 'ndi', 'ccId']
        self.column_reward = ['time', 'cellId', 'IMSI', 'RNTI', 'mcs', 'size', 'rv', 'ndi', 'ccId']

        # self.column_state = ['RNTI', 'mcs', 'size', 'rv', 'ndi', 'ccId']
        # self.column_reward = ['RNTI', 'mcs', 'size', 'rv', 'ndi', 'ccId']

    def read_kpms(self, kpm_type='state', filename="DlTxPhyStats.txt"):
        assert kpm_type in ('state', 'reward')
        columns = self.column_state if kpm_type=='state' else self.column_reward

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))

        try:
            df = pd.read_csv(file_path, delim_whitespace=True, comment='%', names=self.column_state,
                             skiprows=1, index_col=False)
        except Exception as e:
            print(f"Error reading the file {filename}: {e}")
            return pd.DataFrame()

        # Sample rows where time == max(time)
        df_results = df[df['time'] == df['time'].max()][columns].copy()
        df_results = self.filter_df_by_kpms(df_results)
        df_results = df_results.sort_values(by=['cellId', 'IMSI'], ascending=[True, True])

        return df_results

    def filter_df_by_enbs(self, df):
        """
        Filter df by CellId for each eNB
        """
        df_results = pd.DataFrame()

        for idx in range(1, self.num_enbs + 1):  # cellId 1 to num_enbs
            filtered_df = df[df['cellId'] == str(idx)]
            df_results = pd.concat([df_results, filtered_df])

        return df_results

    def filter_df_by_kpms(self, df):
        """
        Compute throughput for required KPMs
        """
        df_copy = df.copy()

        df_copy['throughput'] = df['size']/8

        return df_copy


if __name__ == '__main__':
    class Args:
        time_step = 1  # 1 ms
        num_enbs = 4  # 4 base stations (cell IDs 1 to 4)

    args = Args()
    parser = DataParser(args)

    df_kpms = parser.read_kpms(filename="DlTxPhyStats.txt")
    print(df_kpms)
    print(np.array(df_kpms))

    # Filter data by eNBs
    df_filtered = parser.filter_df_by_enbs(df_kpms)
    print(df_filtered)

    # Compute throughput
    df_with_tp = parser.filter_df_by_kpms(df_kpms)
    print(df_with_tp)
