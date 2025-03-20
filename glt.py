import os
import pandas as pd

from nsoran.data_parser import DataParser

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

class Args:
    def __init__(self):
        self.num_enb = 4

args = Args()
dp = DataParser(args)

kpm_type = 'tp'
columns = dict_columns.get(kpm_type)
usecols = ['time', 'cellId'] + dict_kpms.get(kpm_type)
filename = dict_filenames.get(kpm_type)
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

df = pd.read_csv(file_path, sep='\s+', comment='%', index_col=False, names=columns, skiprows=1, usecols=usecols)
if df['time'].dtype == 'float':
    df['time'] *= 1000
    df['time'] = df['time'].astype(int)
# df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
print(df.tail())
print(df['mcs'].unique())

# kpm_type = 'sinr'
# for kpm_type in dict_kpms.keys():
#     filename = dict_filenames.get(kpm_type)
#     file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
#     try:
#         with open(file_path, 'r') as f:
#             latest_time_str = f.readlines()[3893].strip().split()[0]  # Extract the first value of the last line
#             latest_time_ms = int(float(latest_time_str) * 1000) if '.' in latest_time_str else int(latest_time_str)
#         print(f"\nlatest_time_str: {latest_time_str}, {type(latest_time_str)}")
#         print(f"latest_time_ms: {latest_time_ms}, {type(latest_time_ms)}")
#     except Exception as e:
#         print(e)


    # columns = dict_columns.get(kpm_type)
    # usecols = ['time', 'cellId'] + dict_kpms.get(kpm_type)
    # df = pd.read_csv(file_path, sep='\s+', comment='%', index_col=False, names=columns, skiprows=1, usecols=usecols)
    # # print(df['time'][0], df['time'].dtype == 'float')
    #
    # if df['time'].dtype == 'float':
    #     df['time'] *= 1000
    #     df['time'] = df['time'].astype(int)

