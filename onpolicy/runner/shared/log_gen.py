import os
import time
import log_parser

global last


def tpcalc(df, last):
    tp=[]
    tdf=df.iloc[last:]
    tdf=tdf.loc[tdf['cellId'] == '1']
    tp.append((sum(tdf['size'])/8))

    tdf=df.iloc[last:]
    tdf=tdf.loc[tdf['cellId'] == '2']
    tp.append((sum(tdf['size'])/8))

    tdf=df.iloc[last:]
    tdf=tdf.loc[tdf['cellId'] == '3']
    tp.append((sum(tdf['size'])/8))

    tdf=df.iloc[last:]
    tdf=tdf.loc[tdf['cellId'] == '4']
    tp.append((sum(tdf['size'])/8))

    return tp


def get_data(last,fifo):
    data=os.read(fifo,13)
    print(data)
    try:
        data=data.decode("utf-8")
    except UnicodeDecodeError:
        print("except")
        print(data[0])
        data="60,60,60,60"
    tp = [0,0,0,0]
    tx= [0,0,0,0]
    if(not data):
        return {'txpower' : tx, 'tp':tp}
    data=data.split(',')
    print("Received: ",data)
    if(data):
        t=log_parser.dltxphy()
        # print(" Lines read till now : " + str(last))
        
        if(len(t)):
            tp = tpcalc(t,last)
            last=len(t)
        print("Do something \n")
        
        print(data)
        print(tp)
        inpt={'txpower' : data, 'tp':tp}
        print(inpt)
        return inpt,last
    
def send_action(txp,fifo2):
    txp="0,"+txp
    print("sending action")
    os.write(fifo2,txp.encode("utf-8"))


# if os.path.isfile("DlRxPhyStats.txt"):
#     print("File deleted \n")
#     os.remove("DlRxPhyStats.txt")
# if os.path.exists("fifo2"):
#         os.unlink("fifo2")

# os.mkfifo("fifo2")
# fifo=os.open("fifo1",os.O_RDONLY)
# fifo2=os.open("fifo2", os.O_WRONLY)
# global last
# print("Opening FIFO...")

# last=0   

# while True:
#     data = get_data(last)
#     if(data==[0,0,0,0]):
#         continue
#     txp="65,75,80,90 ,\0"
#     send_action(txp)









    # data=os.read(fifo,15)
    # print(data)
    # data=data.decode("utf-8")
    


    # if(not data):
    #     continue
    # print("Received: ",data)

    # t=log_parser.dltxphy()
    # # print(t)
    # if(len(t)):
    #     print(tpcalc(t,last))
    #     last=len(t)
    
    # print("Do something \n")
    # #generate_log()
    # os.write(fifo2,"1".encode("utf-8"))