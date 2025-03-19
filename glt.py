import os
import pandas as pd

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

kpm_type = 'sinr'

filename = dict_filenames.get(kpm_type)
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
try:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip().split()
        print(f"last_line: {last_line}")
        print(f"time: {last_line[0]}, {type(last_line[0])}")
except Exception as e:
    print(e)


columns = dict_columns.get(kpm_type)
usecols = ['time', 'cellId'] + dict_kpms.get(kpm_type)
df = pd.read_csv(file_path, sep='\s+', comment='%', index_col=False, names=columns, skiprows=1, usecols=usecols)
print(df['time'][0], df['time'].dtype == 'float')

if df['time'].dtype == 'float':
    df['time'] *= 1000
    df['time'] = df['time'].astype(int)
# print(df[-30:])
