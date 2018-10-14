import pandas as pd
import numpy as np

dir="F:/研究生课题/##小论文写作/数据挖掘论文/Python_Project/Processed_Data"
#reside=["J1","J2","J3","J4","J5","J6","J7","Z1","Z2","Z3","Z4","Z5","Z6"]
reside=["J1"]
month=["02","03","04","05","06","07","08","09","10","11","12"]
for res in reside:
    res_data=pd.DataFrame([])
    for m in month:
        directory=dir+"/"+res+"/"+res+"_"+str(m)+".xlsx"
        print(directory)
        data=pd.read_excel(directory,sheet_name=None)
        ik=data['BR'].iloc[:,0:6]
        TimeSeries=pd.to_datetime(data['BR'].iloc[:,0], format="%Y/%m/%d %H:%M:%S")
        TimeSeries=pd.DatetimeIndex(TimeSeries)
        month=TimeSeries.month
        week=TimeSeries.weekday
        hour=TimeSeries.hour
        month=pd.Series(month)
        week=pd.Series(week)
        hour=pd.Series(hour)
        timefactor={"Month":month,"Week":week, "Hour":hour }
        timefactor=pd.DataFrame(timefactor)
        ik=pd.merge(timefactor,ik,left_index=True,right_index=True)
        wt=data['WT'].iloc[:,1:4]
        wt = pd.DataFrame(wt.values, columns=wt.columns)
        BRW=data['BRWin'].iloc[:,0:1]
        BRW = pd.DataFrame(BRW.values, columns=BRW.columns)
        PM=data['PM'].iloc[:,1:]
        PM = pd.DataFrame(PM.values, columns=PM.columns)
        ikwt=pd.merge(ik,wt,left_index=True,right_index=True)
        PMBRW=pd.merge(PM,BRW,left_index=True,right_index=True)
        merged_data=pd.merge(ikwt,PMBRW,left_index=True,right_index=True,how="inner")
        res_data=pd.concat((res_data,merged_data), ignore_index=True)
        print(res_data)
    res_data.to_csv(dir + "/" + res + "/" + res + "_annually_data.csv")