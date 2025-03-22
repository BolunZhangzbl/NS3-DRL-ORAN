# -- Public Imports
import os
import time
import threading
import numpy as np
import pandas as pd

# -- Private Imports
from nsoran.utils import *

# -- Global Variables

dict_kpms = dict(
    tp=['size', 'mcs'],
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


# -- Functions

class DataParser:

    def __init__(self, args):
        self.last_read_time = 0
        self.num_enb = args.num_enb

    def get_latest_time(self, kpm_type):
        """Get the latest timestamp for a KPM file"""
        filename = dict_filenames.get(kpm_type)
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))

        try:
            with open(file_path, 'r') as f:
                latest_time_str = f.readlines()[-1].strip().split()[0]  # Extract the first value of the last line
                latest_time_float = float(latest_time_str)
                if '.' in latest_time_str or latest_time_float<10:
                    latest_time_ms = int(latest_time_float * 1000)
                else:
                    latest_time_ms = int(latest_time_str)
                return latest_time_ms
        except Exception as e:
            print(f"Error reading latest time for {kpm_type}: {e}")
            return self.last_read_time

    def read_kpms(self, kpm_type, start_time, end_time):
        """Read data for a specific KPM type within [start_time, end_time]."""
        columns = dict_columns.get(kpm_type)
        usecols = ['time', 'cellId'] + dict_kpms.get(kpm_type)
        filename = dict_filenames.get(kpm_type)
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))

        try:
            df = pd.read_csv(file_path, sep='\s+', comment='%', index_col=False, names=columns, skiprows=1, usecols=usecols)
            df = self.convert_time_to_ms(df)  # Your existing time conversion
            df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        except Exception as e:
            print(f"Error reading {kpm_type}: {e}")
            return pd.DataFrame()

        # Process KPM-specific logic (groupby, rename, etc.)
        if kpm_type == 'tp':
            df['mcs'] = df['mcs'].astype(int)
            df['mcs_bits'] = df['mcs'].apply(map_mcs_bits)
            df['cr'] = df['mcs'].apply(mcs_to_cr_func)
            df['prbs'] = (df['size'] * 8) / (df['mcs_bits'] * 84 * df['cr'])
            df['tp'] = df['size'] * 8
            df = df.drop(columns=['mcs', 'mcs_bits', 'cr', 'size'], errors='ignore')
            df = df.groupby('cellId', as_index=False).mean()

        elif kpm_type == 'sinr':
            df = df.groupby('cellId', as_index=False).mean()

        return df.drop(columns=['time'], errors='ignore')

    def aggregate_kpms(self):
        # Step 1: Determine the common time window
        end_time_tp = self.get_latest_time('tp')
        end_time_sinr = self.get_latest_time('sinr')
        end_time = min(end_time_tp, end_time_sinr)

        # Ensure we ignore the last file time idx affect
        end_time = min(end_time, self.last_read_time + 200)

        # Swap end_time and last_read_time if
        if end_time < self.last_read_time:
            end_time, self.last_read_time = self.last_read_time, end_time

        # Step 2: Read all KPMs within [last_read_time, end_time]
        df_tp = self.read_kpms('tp', self.last_read_time, end_time)
        df_sinr = self.read_kpms('sinr', self.last_read_time, end_time)

        # Step 3: Merge data and fill missing cellIds
        df_aggregated = df_tp.merge(df_sinr, on='cellId', how='outer')
        df_aggregated = self.fill_missing_cellid(df_aggregated)  # Your existing method

        # Step 4: Update last_read_time AFTER processing all files
        print(f"self.last_read_time: {self.last_read_time}   end_time: {end_time}\n")
        self.last_read_time = end_time

        return df_aggregated

    def convert_time_to_ms(self, df):
        """
        Convert time column to milliseconds if it is in seconds.
        """
        # Calculate the range of the time column
        if df['time'].dtype == 'float':
            df['time'] *= 1000
            df['time'] = df['time'].astype(int)
        return df

    def fill_missing_cellid(self, df):
        """
        Ensure all expected cellIds are present, filling missing rows with 0 for numerical columns.
        Maintains original data types for non-numeric columns.
        """
        # Convert cellId to integer and get expected IDs
        df['cellId'] = df['cellId'].astype(int)
        expected_cell_ids = range(1, self.num_enb + 1)

        # Create a DataFrame with all expected cellIds
        full_cell_ids = pd.DataFrame({'cellId': expected_cell_ids})

        # Merge to add missing cellIds (left join on full_cell_ids)
        df = full_cell_ids.merge(df, on='cellId', how='left')

        # Identify numerical columns to fill with 0
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove('cellId')  # Skip cellId (it's already correct)

        # Fill numerical columns with 0 where missing
        df[numerical_cols] = df[numerical_cols].fillna(0)

        # Sort and reset index
        return df.sort_values('cellId').reset_index(drop=True)

# class DataParser:
#
#     def __init__(self, args):
#
#         self.time_step = 0
#         self.it_period = args.it_period   # in ms
#         self.num_enb = args.num_enb
#         self.last_read_time = 0        # Store the last processed timestamp
#         self.latest_time = 0
#
#     def read_kpms(self, kpm_type):
#         assert kpm_type in dict_kpms
#
#         columns = dict_columns.get(kpm_type)
#         usecols = ['time', 'cellId'] + dict_kpms.get(kpm_type)
#         filename = dict_filenames.get(kpm_type)
#
#         file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", filename))
#
#         try:
#             # Read the CSV file
#             df = pd.read_csv(file_path, sep='\s+', comment='%', names=columns,
#                              skiprows=1, index_col=False, usecols=usecols)
#             self.time_step += 1
#             print(f"Successfully read the file {filename}")
#         except Exception as e:
#             print(f"Error reading the file {filename}: {e}")
#             return pd.DataFrame(), None
#
#         # Parse time col - Ensure conversion is working
#         df = self.convert_time_to_ms(df)
#
#         # Only keep rows after last_read_time
#         if self.time_step%1==0 and self.time_step>0:
#             latest_time = df['time'].loc[-1][0]
#             self.latest_time = latest_time
#         else:
#             if self.latest_time > self.last_read_time:
#                 df = df[df['time'] >= self.last_read_time]
#
#         # Update last_read_time for 'prb'
#         if self.time_step%3==0 and self.time_step>0:
#             print(f"latest_time: {self.latest_time}")
#             self.last_read_time = self.latest_time
#
#         # If DataFrame is empty, return early
#         if df.empty:
#             print(f"Warning: DataFrame is empty after filtering, returning empty DataFrame.")
#             return pd.DataFrame()
#
#         # Drop 'time' column
#         df = df.drop(columns=['time'], errors='ignore')
#
#         # Process different kpm types
#         if kpm_type == 'tp':
#             df = df.groupby('cellId', as_index=False).mean()
#             df['size'] = df['size'] / 8
#             df = df.rename(columns={'size': 'tp'})
#         elif kpm_type == 'sinr':
#             df = df.groupby('cellId', as_index=False).mean()
#         else:
#             df = df.groupby('cellId', as_index=False).mean()
#             df = df.rename(columns={'sizeTb1': 'prb'})
#
#         return df
#
#     def aggregate_kpms(self):
#         df_tp = self.read_kpms(kpm_type='tp')
#         df_sinr = self.read_kpms(kpm_type='sinr')
#         df_prb = self.read_kpms(kpm_type='prb')
#
#         print(f"self.last_read_time: {self.last_read_time}")
#         # print("df_tp columns:", df_tp.columns)
#         # print("df_sinr columns:", df_sinr.columns)
#         # print("df_prb columns:", df_prb.columns)
#
#         df_aggregated = df_tp.merge(df_sinr, on='cellId', how='outer').merge(df_prb, on='cellId', how='outer')
#         df_aggregated = self.fill_missing_cellid(df_aggregated)
#
#         return df_aggregated
#
#     def convert_time_to_ms(self, df):
#         """
#         Convert time col to ms if its in sec
#         """
#         if df['time'].max() < 1000:
#             df['time'] *= 1000
#             print(f"Convert sec to ms!")
#         else:
#             print("Already in ms!")
#
#         return df
#
#     def fill_missing_cellid(self, df):
#         """
#         Fill values for missing cellId with rows containing zeros for all columns.
#         """
#         # Ensure cellId is of integer type
#         df['cellId'] = df['cellId'].astype(int)
#
#         # Expected cell IDs
#         expected_cell_ids = set(range(1, self.num_enb+1))
#
#         # Actual cell IDs in the DataFrame
#         actual_cell_ids = set(df['cellId'].unique())
#
#         # Check if all expected cell IDs are present
#         if actual_cell_ids == expected_cell_ids:
#             return df
#
#         # Find missing cell IDs
#         missing_cell_ids = expected_cell_ids - actual_cell_ids
#
#         # Create a DataFrame for missing rows
#         if missing_cell_ids:
#             missing_data = {col: 0 for col in df.columns}  # Initialize with zeros
#             missing_df = pd.DataFrame([missing_data] * len(missing_cell_ids))  # Create multiple rows
#             missing_df['cellId'] = list(missing_cell_ids)  # Assign correct cellId values
#
#             # Concatenate the original DataFrame with the missing rows
#             df = pd.concat([df, missing_df], ignore_index=True)
#
#         # Sort the DataFrame by cellId for consistency
#         df = df.sort_values(by='cellId').reset_index(drop=True)
#
#         return df



# def test_fifo():
#     class Args:
#         it_period = 1  # 1 ms
#         num_enb = 4  # 4 base stations (cell IDs 1 to 4)
#
#     args = Args()
#     parser = DataParser(args)
#
#     df_kpms = parser.aggregate_kpms()
#     print(df_kpms)  # ✅ First print (no blocking)
#
#     fifo_path = "/tmp/test_fifo"
#
#     # Remove old FIFO if it exists
#     if os.path.exists(fifo_path):
#         os.remove(fifo_path)
#
#     # ✅ Create FIFO
#     os.mkfifo(fifo_path)
#
#     # ✅ Function to read FIFO (Runs in a separate thread)
#     def read_fifo():
#         with open(fifo_path, "r") as fifo:
#             data = fifo.read().strip()  # Read all at once
#             print("Read from FIFO:", data)
#             data_tx_power = list(map(int, data.split(",")))
#             print("Parsed Tx power:", data_tx_power)
#
#             # ✅ Add tx_power column
#             df_kpms['tx_power'] = data_tx_power
#             print(df_kpms)
#
#     # ✅ Start reader in a separate thread (to prevent blocking)
#     read_thread = threading.Thread(target=read_fifo)
#     read_thread.start()
#
#     # ✅ Small delay to ensure the reader is ready
#     time.sleep(0.5)
#
#     # ✅ Write to FIFO (Runs in main thread)
#     with open(fifo_path, "w") as fifo:
#         test_data = "10,20,30,40\n"  # Example Tx power values
#         fifo.write(test_data)
#
#     # ✅ Wait for the reader thread to finish
#     read_thread.join()
#
#
# if __name__ == '__main__':
#
#     test_fifo()
