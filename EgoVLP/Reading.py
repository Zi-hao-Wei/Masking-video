import os
import pandas as pd
import json

with open(r".\test_videodatainfo.json") as f:
    data = json.load(f)

# for k in data["sentences"]:
    # print(k)
data = pd.json_normalize(data["sentences"])
# print(df)
# data = pd.read_csv("./MSVD_train.csv",sep=",",on_bad_lines="warn")
# # data["path"]="./YouTubeClips/"+data["VideoID"]+"_"+data["Start"].astype("str")+"_"+data["End"].astype("str")+".avi"
# print(data)
# data["VideoID"]=data["video_id"]
l1 = list(data.groupby("video_id"))
smaller = []
for i in l1:
    # print(type(i))
    tmp = i[1].head(1)
    # print(smaller)
    smaller.append(tmp)
new = pd.concat(smaller)
# new = new.drop(columns=new.columns[0], axis=1)
# print(new)
new.to_csv("MSRVTT_test.csv")
#     # exit()
# print(data.__len__())
# for i in range(data.__len__()):
#     # print(i)
#     path = data["path"].iloc[i]
#     if not (os.path.exists(path)):
#         print(path)