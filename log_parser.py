import os 
import pandas as pd


def dlsinr():
    phy = pd.DataFrame(columns=['time', 'cellId', 'IMSI', 'RNTI', 'rsrp', 'sinr', 'ComponentCarrierId'])

    if(os.path.exists('DlRsrpSinrStats.txt')):
        f=open('DlRsrpSinrStats.txt','r')
        lines=f.readlines()
        f.close()
        for i in lines:
            i=i.split("\t")
            if(i[0]=='% time'):
                continue
            else:
                pd.concat([phy, pd.DataFrame([[i[0], i[1], i[2], i[3], i[4], i[5], i[6]]], columns=['time', 'cellId', 'IMSI', 'RNTI', 'rsrp', 'sinr', 'ComponentCarrierId'])], ignore_index=True)
                
    else:
        return
    return phy

def dltxphy(last):
    tx = pd.DataFrame(columns = ['% time',	'cellId',	'IMSI',	'RNTI',	'txMode',	'layer',	'mcs',	'size'])
    if(os.path.exists('/home/waleed/holistic/DlRxPhyStats.txt')):
        f=open('/home/waleed/holistic/DlRxPhyStats.txt', 'r')
        lines = f.readlines();
        f.close()
        count=-1
        for i in lines:
            if(count!=last):
                count+=1
                continue    
            i=i.split('\t')
            
            if(i[0] == '% time'): 
                continue
            else:
                tx=(pd.concat([tx,pd.DataFrame([[i[0], i[1], i[2], i[3], i[4], i[5], i[6], int(i[7])]], columns = ['% time',	'cellId',	'IMSI',	'RNTI',	'txMode',	'layer',	'mcs',	'size'])], ignore_index=True))
    else :
        # print("File not found")
        return tx
    return tx

def tpcalc(df):
    tp=[]
    tdf=df
    tdf=tdf.loc[tdf['cellId'] == '1']
    tp.append((sum(tdf['size'])/8))

    tdf=df
    tdf=tdf.loc[tdf['cellId'] == '2']
    tp.append((sum(tdf['size'])/8))

    tdf=df
    tdf=tdf.loc[tdf['cellId'] == '3']
    tp.append((sum(tdf['size'])/8))

    tdf=df
    tdf=tdf.loc[tdf['cellId'] == '4']
    tp.append((sum(tdf['size'])/8))

    return tp

