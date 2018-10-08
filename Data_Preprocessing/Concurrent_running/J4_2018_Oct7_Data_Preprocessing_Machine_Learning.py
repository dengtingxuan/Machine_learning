# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import re
import os
import numpy as np

# month="03"
reside = ["J4"]
month = ["12"]
# reside=["J1","J2","J3","J4","J5","J6","J7","Z1","Z2","Z3","Z4","Z5","Z6"]
# month=["02","03","04","05","06","07","08","09","10","11","12"]
month_day = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

dictmonth = dict()
i = 0
for m in month:
    dictmonth[m] = month_day[i]
    i = i + 1

for res in reside:
    for m in month:
        if int(m) < 10:
            a = str(m)
            m1 = a[- 1]
        else:
            m1 = m
        dire = 'F:\十三五课题/135Data\Raw_Data\KM_Data\KM_M{}/KM{}_M_2017-{}.xlsx'.format(m1, res, m)
        A = pd.ExcelFile(dire)
        Sheetname = A.sheet_names
        print(Sheetname)
        # print(a)
        for i in range(len(Sheetname)):
            if (not re.search('卧', Sheetname[i]) == None) and (re.search('窗', Sheetname[i]) == None) and (
                    re.search('空', Sheetname[i]) == None) and (re.search('风', Sheetname[i]) == None):
                BRindex = i
            if (not re.search('客', Sheetname[i]) == None) and (re.search('窗', Sheetname[i]) == None) and (
                    re.search('空', Sheetname[i]) == None) and (re.search('风', Sheetname[i]) == None):
                LRindex = i
            if (not re.search('客', Sheetname[i]) == None) and (not re.search('窗', Sheetname[i]) == None):
                LRWindex = i
            if (not re.search('卧', Sheetname[i]) == None) and (not re.search('窗', Sheetname[i]) == None):
                BRWindex = i
            if (not re.search('卧', Sheetname[i]) == None) and (not re.search('门', Sheetname[i]) == None):
                BRDindex = i
            if (not re.search('天气', Sheetname[i]) == None):
                weather = i
            if (not re.search('pm', Sheetname[i]) == None):
                pm = i

        BR_data = pd.read_excel(dire, sheetname=BRindex, parse_cols=[9, 10, 11, 12, 13, 15], skiprows=[0])
        BR_data.columns = ["Time", "Temperature", "RH", "HCHO", "CO2", "PM2.5"]
        BR_data["Time"] = pd.to_datetime(BR_data["Time"], format="%Y/%m/%d %H:%M:%S")

        Weather_data = pd.read_excel(dire, sheetname=weather, parse_cols=[8, 9, 10, 11, 12, 13], skiprows=[0])
        Weather_data.columns = ["Time", "Out_Temperature", "Out_RH", "Rain", "Wind_V", "Wind_D"]
        Weather_data["Time"] = pd.to_datetime(Weather_data["Time"], format="%Y/%m/%d %H:%M:%S")
        # Weather_add=globals()
        # print(Weather_data)
        # print(Weather_data.iloc[j,0])
        # for j in range(len(Weather_data)):
        #     Tw=Weather_data.iloc[j,0]
        #     for Tbr in BR_data["Time"]:
        #         if Tbr-Tw < datetime.timedelta(minutes=120):
        #             Weather_data=pd.concat((Weather_data,Weather_data.iloc[j,1:5]))
        #             print(Weather_data.iloc[j,1:5])
        #         else:
        #             Weather_data = pd.concat((Weather_data, Weather_data.iloc[j+1, 1:5]))
        #             break
        Weather_add = pd.DataFrame(columns=["Time", "Out_Temperature", "Out_RH", "Rain", "Wind_V", "Wind_D"])
        #
        i = 0
        s = 0
        start_date = pd.to_datetime("2017-{}-01 00:00:00".format(m))
        end_date = start_date + datetime.timedelta(hours=2)

        # and (i != A[-19])
        while (i < len(BR_data) and end_date <= pd.to_datetime("2017-{}-{} 23:59:59".format(m, dictmonth[m]))):
            # A.append(i)
            mask = (BR_data["Time"] >= start_date) & (BR_data["Time"] < end_date)
            lenth = len(BR_data.loc[mask])  # 选取IAQ的一段时间
            i = i + lenth
            # print(BR_data.loc[mask])
            counter = 0
            # print(len(Weather_data))
            for s in range(len(Weather_data)):  # 对每个天气里面的时间作检索
                Tw = Weather_data.iloc[s, 0]  # 选取时间
                counter = counter + 1
                if ((Tw - start_date <= datetime.timedelta(minutes=2)) and (
                        Tw - start_date >= -datetime.timedelta(minutes=2))):
                    # 条件：当选取的天气时间距离时间块的首行小于1h
                    print(start_date)
                    # print("True")
                    # print(lenth)
                    for p in range(lenth):  # 那么叠加length个天气块
                        # print((Weather_data.iloc[s,0:5]))
                        # print((Weather_add))
                        X = Weather_data.iloc[s, 0:6]
                        X = X.transpose()
                        Weather_add = Weather_add.append(X)
                        # print(Weather_data.iloc[s,0:5])
                    print(Weather_add)
                    break
                if counter == len(Weather_data):
                    for p in range(lenth):  # 那么叠加length个天气块
                        ROW = pd.DataFrame(["NA", "NA", "NA", "NA", "NA", "NA"],
                                           index=["Time", "Out_Temperature", "Out_RH", "Rain", "Wind_V", "Wind_D"],
                                           columns=["N"])
                        ROW = ROW.transpose()
                        Weather_add = Weather_add.append(ROW)
            start_date = start_date + datetime.timedelta(hours=2)
            end_date = start_date + datetime.timedelta(hours=2)

            print(len(BR_data))
            print("i equals:", i)
            # print(Weather_add)

        PM_add = pd.DataFrame(
            columns=["Time", "CO(mg/m3)", "NO2(ug/m3)", "SO2(ug/m3)", "O3(ug/m3)", "PM10(ug/m3)", "PM2.5(ug/m3)",
                     "AQI"])
        #
        j = 0
        s = 0
        start_date = pd.to_datetime("2017-{}-01 00:00:00".format(m))
        end_date = start_date + datetime.timedelta(hours=2)

        PM_data = pd.read_excel(dire, sheetname=pm, parse_cols=[8, 9, 10, 11, 12, 13, 14, 15], skiprows=[0])
        PM_data.columns = ["Time", "CO(mg/m3)", "NO2(ug/m3)", "SO2(ug/m3)", "O3(ug/m3)", "PM10(ug/m3)", "PM2.5(ug/m3)",
                           "AQI"]
        PM_data["Time"] = pd.to_datetime(PM_data["Time"], format="%Y/%m/%d %H:%M:%S")

        while (j < len(BR_data) and end_date <= pd.to_datetime("2017-{}-{} 23:59:59".format(m, dictmonth[m]))):
            # A.append(i)
            mask = (BR_data["Time"] >= start_date) & (BR_data["Time"] < end_date)
            lenth = len(BR_data.loc[mask])  # 选取IAQ的一段时间
            j = j + lenth
            # print(BR_data.loc[mask])
            counter = 0
            # print(len(Weather_data))
            for s in range(len(PM_data)):  # 对每个天气里面的时间作检索
                Tw = PM_data.iloc[s, 0]  # 选取时间
                counter = counter + 1
                if ((Tw - start_date <= datetime.timedelta(minutes=2)) and (
                        Tw - start_date >= -datetime.timedelta(minutes=2))):
                    # 条件：当选取的天气时间距离时间块的首行小于1h
                    print(start_date)
                    # print("True")
                    # print(lenth)
                    for p in range(lenth):  # 那么叠加length个室外空气块
                        # print((Weather_data.iloc[s,0:5]))
                        # print((Weather_add))
                        X = PM_data.iloc[s, 0:8]
                        X = X.transpose()
                        PM_add = PM_add.append(X)
                        # print(Weather_data.iloc[s,0:5])
                    print(PM_add)
                    break
                if counter == len(PM_data):
                    for p in range(lenth):  # 那么叠加length个室外空气质量块
                        ROW = pd.DataFrame(["NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"],
                                           index=["Time", "CO(mg/m3)", "NO2(ug/m3)", "SO2(ug/m3)", "O3(ug/m3)",
                                                  "PM10(ug/m3)", "PM2.5(ug/m3)", "AQI"], columns=["N"])
                        ROW = ROW.transpose()
                        PM_add = PM_add.append(ROW)
            start_date = start_date + datetime.timedelta(hours=2)
            end_date = start_date + datetime.timedelta(hours=2)
            print(len(BR_data))
            print("j equals:", j)
            # print(Weather_add)

        BRWin_data = pd.read_excel(dire, sheetname=BRWindex, parse_cols=[11, 12], skiprows=[0])
        BRWin_data.columns = ["Time", "BRW"]
        BRWin_data["Time"] = pd.to_datetime(BRWin_data["Time"], format="%Y/%m/%d %H:%M:%S")
        DA = pd.DataFrame(columns=["BRW"])

        for bl in range(len(BRWin_data["BRW"])):
            if BRWin_data.iloc[bl, 1] == "open":
                a = int(1)
                BRWin_data.iloc[bl, 1] = a
            else:
                b = int(0)
                BRWin_data.iloc[bl, 1] = b
        print(BRWin_data)
        BRWin_add = pd.DataFrame([])
        # BRWin_add.columns=["Time","Status"]
        # print(BRWin_data.iloc[1,0])
        # print(len(BRWin_data))

        deltatime = BRWin_data.iloc[0, 0]
        mask0 = (BR_data["Time"] >= ("2017-{}-01 00:00:00".format(m))) & (BR_data["Time"] < deltatime)
        lenth0 = len(BR_data.loc[mask0])  # 选取IAQ的一段时间
        print(lenth0)
        for q in range(lenth0):
            Modif = BRWin_data.iloc[0, 0:2]
            if Modif[1] == 1:
                Modif[1] = 0
            else:
                Modif[1] = 1
            X = Modif
            X = X.transpose()
            BRWin_add = BRWin_add.append(X)

        for i in range(len(BRWin_data) - 1):
            print(BRWin_data.iloc[i, 0])
            print(BRWin_data.iloc[i + 1, 0])
            deltatime = BRWin_data.iloc[i, 0]
            deltatime1 = BRWin_data.iloc[i + 1, 0]
            mask = (BR_data["Time"] >= deltatime) & (BR_data["Time"] < deltatime1)
            lenth = len(BR_data.loc[mask])  # 选取IAQ的一段时间
            print(lenth)
            for p in range(lenth):
                X = BRWin_data.iloc[i, 0:2]
                X = X.transpose()
                BRWin_add = BRWin_add.append(X)
        print(BRWin_add)

        # comb=pd.concat([BR_data,Weather_add,PM_add])
        writer = pd.ExcelWriter("F:\研究生课题\##小论文写作\数据挖掘论文\Python_Project\Processed_Data/{}_{}.xlsx".format(res, m))
        BR_data.to_excel(writer, "BR")
        Weather_add.to_excel(writer, "WT")
        PM_add.to_excel(writer, "PM")
        BRWin_add.to_excel(writer, "BRWin")
        writer.save()
        writer.close()

# PM_add=pd.DataFrame(columns =["Time","CO(mg/m3)","NO2(ug/m3)","SO2(ug/m3)","O3(ug/m3)","PM10(ug/m3)","PM2.5(ug/m3)","AQI"])
# start_date = pd.to_datetime("2017-{}-01 00:00:00".format(month))
# end_date = start_date+datetime.timedelta(hours=2)
# i=0
# s=0
# while i <len(BR_data):
#     mask=(BR_data["Time"]>=start_date) & (BR_data["Time"]<end_date)
#     lenth=len(BR_data.loc[mask])# 选取IAQ的一段时间
#     #print(BR_data.loc[mask])
#     counter=0
#     print(len(PM_data))
#     for s in range(len(PM_data)): #对每个天气里面的时间作检索
#         Tw=PM_data.iloc[s,0]#选取时间
#         counter = counter + 1
#         if ((Tw-start_date<= datetime.timedelta(minutes=2)) and (Tw-start_date>=-datetime.timedelta(minutes=2))):
#             #条件：当选取的天气时间距离时间块的首行小于1h
#             print(start_date)
#             #print("True")
#             #print(lenth)
#             for p in range(lenth):#那么叠加length个天气块
#                 #print((Weather_data.iloc[s,0:5]))
#                 #print((Weather_add))
#                 X=PM_data.iloc[s, 0:7]
#                 X=X.transpose()
#                 PM_add=PM_add.append(X)
#                 #print(Weather_data.iloc[s,0:5])
#             print(PM_add)
#             break
#         if counter == len(PM_data):
#              for p in range(lenth):  # 那么叠加length个天气块
#                  ROW = pd.DataFrame(["NA","NA","NA","NA","NA","NA","NA","NA"],index=["Time","CO(mg/m3)","NO2(ug/m3)","SO2(ug/m3)","O3(ug/m3)","PM10(ug/m3)","PM2.5(ug/m3)","AQI"],columns=["N"])
#                  ROW=ROW.transpose()
#                  PM_add = PM_add.append(ROW)
#     start_date=start_date+datetime.timedelta(hours=2)
#     end_date=start_date+datetime.timedelta(hours=2)
#     i=i+lenth
#     print(i)
#
# writer=pd.ExcelWriter("C:/Users\dengtingxuan\Desktop/PM_Data.xlsx")
# PM_add.to_excel(writer,"PM")
# writer.save()
# writer.close()
#
#
#
# WT_PM=pd.concat((Weather_add,PM_add),ignore_index=True)
# writer=pd.ExcelWriter("C:/Users\dengtingxuan\Desktop/WT_PM_Data.xlsx")
# WT_PM.to_excel(writer,"WT_PM")
# writer.save()
# writer.close()
