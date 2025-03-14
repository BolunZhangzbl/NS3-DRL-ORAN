# -- Public Imports
import os
import time
import threading
import numpy as np
import pandas as pd

# -- Private Imports
from utils import *

# -- Global Variables

dict_kpms = dict(
    tp=['size'],
    sinr=['sinr'],
    prb=['sizeTb1'],
)

dict_columns = dict(
    tp=["time", "cellId", "IMSI", "RNTI", "txMode", "layer", "mcs", "size", "rv", "ndi", "correct", "ccId"],
    sinr=['time', 'cellId', 'IMSI', 'RNTI', 'rsrp', 'sinr', 'ComponentCarrierId'],
    prb=['time', 'cellId', 'IMSI', 'frame', 'sframe', 'RNTI', 'mcsTb1', 'sizeTb1', 'mcsTb2', 'sizeTb2', 'ccId']
)


dict_filenames = dict(
    tp='DlRxPhyStats.txt',
    sinr='DlRsrpSinrStats.txt',
    prb='DlMacStats.txt'
)

dict_fail = dict(
    tp=[0],
    sinr=[-10],
    prb=[0]
)


# -- Functions

class DataParser:

    def __init__(self, args):

        self.time_step = args.time_step   # in ms
        self.num_enb = args.num_enb
        self.last_read_time = None        # Store the last processed timestamp

    def read_kpms(self, kpm_type):
        assert kpm_type in dict_kpms

        columns = dict_columns.get(kpm_type)
        usecols = ['time', 'cellId'] + dict_kpms.get(kpm_type)
        filename = dict_filenames.get(kpm_type)

        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))

        try:
            df = pd.read_csv(file_path, delim_whitespace=True, comment='%', names=columns,
                             skiprows=1, index_col=False, usecols=usecols)
        except Exception as e:
            print(f"Error reading the file {filename}: {e}")
            return pd.DataFrame(), None

        # Parse time col
        # We only set self.last_read_time once, e.g., tp
        df = self.convert_time_to_ms(df)
        if self.last_read_time:
            df = df[df['time'] > self.last_read_time]
        if kpm_type == 'prb':
            latest_time = df['time'].max()
            self.last_read_time = latest_time

        if df.empty:
            return pd.DataFrame()

        df = df.drop(columns=['time'], errors='ignore')

        # Filter by cellId
        if kpm_type == 'tp':
            df = df.groupby('cellId', as_index=False).sum()
            df['size'] = df['size'] / 8
            df = df.rename(columns={'size': 'tp'})
        elif kpm_type == 'sinr':
            df = df.groupby('cellId', as_index=False).mean()
        else:
            df = df.groupby('cellId', as_index=False).sum()
            df = df.rename(columns={'sizeTb1': 'prb'})

        return df

    def aggregate_kpms(self):
        df_tp = self.read_kpms(kpm_type='tp')
        df_sinr = self.read_kpms(kpm_type='sinr')
        df_prb = self.read_kpms(kpm_type='prb')

        df_aggregated = df_tp.merge(df_sinr, on='cellId', how='outer').merge(df_prb, on='cellId', how='outer')
        df_aggregated = self.fill_missing_cellid(df_aggregated)

        return df_aggregated

    def convert_time_to_ms(self, df):
        """
        Convert time col to ms if its in sec
        """
        if df['time'].max() < 1000:
            df['time'] *= 1000
            print(f"Convert sec to ms!")
        else:
            print("Already in ms!")

        return df

    def fill_missing_cellid(self, df):
        """
        Fill values for missing cellId with rows containing zeros for all columns.
        """
        # Ensure cellId is of integer type
        df['cellId'] = df['cellId'].astype(int)

        # Expected cell IDs
        expected_cell_ids = set(range(1, self.num_enb+1))

        # Actual cell IDs in the DataFrame
        actual_cell_ids = set(df['cellId'].unique())

        # Check if all expected cell IDs are present
        if actual_cell_ids == expected_cell_ids:
            return df

        # Find missing cell IDs
        missing_cell_ids = expected_cell_ids - actual_cell_ids

        # Create a DataFrame for missing rows
        if missing_cell_ids:
            missing_data = {col: 0 for col in df.columns}  # Initialize with zeros
            missing_df = pd.DataFrame([missing_data] * len(missing_cell_ids))  # Create multiple rows
            missing_df['cellId'] = list(missing_cell_ids)  # Assign correct cellId values

            # Concatenate the original DataFrame with the missing rows
            df = pd.concat([df, missing_df], ignore_index=True)

        # Sort the DataFrame by cellId for consistency
        df = df.sort_values(by='cellId').reset_index(drop=True)

        return df



def test_fifo():
    class Args:
        time_step = 1  # 1 ms
        num_enb = 4  # 4 base stations (cell IDs 1 to 4)

    args = Args()
    parser = DataParser(args)

    df_kpms = parser.aggregate_kpms()
    print(df_kpms)  # ✅ First print (no blocking)

    fifo_path = "/tmp/test_fifo"

    # Remove old FIFO if it exists
    if os.path.exists(fifo_path):
        os.remove(fifo_path)

    # ✅ Create FIFO
    os.mkfifo(fifo_path)

    # ✅ Function to read FIFO (Runs in a separate thread)
    def read_fifo():
        with open(fifo_path, "r") as fifo:
            data = fifo.read().strip()  # Read all at once
            print("Read from FIFO:", data)
            data_tx_power = list(map(int, data.split(",")))
            print("Parsed Tx power:", data_tx_power)

            # ✅ Add tx_power column
            df_kpms['tx_power'] = data_tx_power
            print(df_kpms)

    # ✅ Start reader in a separate thread (to prevent blocking)
    read_thread = threading.Thread(target=read_fifo)
    read_thread.start()

    # ✅ Small delay to ensure the reader is ready
    time.sleep(0.5)

    # ✅ Write to FIFO (Runs in main thread)
    with open(fifo_path, "w") as fifo:
        test_data = "10,20,30,40\n"  # Example Tx power values
        fifo.write(test_data)

    # ✅ Wait for the reader thread to finish
    read_thread.join()


if __name__ == '__main__':
    # class Args:
    #     time_step = 1  # 1 ms
    #     num_enb = 4  # 4 base stations (cell IDs 1 to 4)
    #
    # args = Args()
    # parser = DataParser(args)
    #
    # df_kpms = parser.aggregate_kpms()
    # print(df_kpms)
    test_fifo()
