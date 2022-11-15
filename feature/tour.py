from haversine import haversine
from tqdm import tqdm
import pandas as pd

tour = pd.read_csv("./jeju_data/tour.csv")
train = pd.read_parquet("./jeju_data/train_new.parquet")
locdf = train.groupby(['end_latitude', 'end_longitude']
                      ).agg({'id': 'count'}).reset_index()

arr = []
for i in tqdm(range(len(locdf))):
    cnt = 0
    my_location = (locdf.iloc[i][0], locdf.iloc[i][1])
    for j in range(0, len(tour)):
        tour_place = (tour.iloc[j][1], tour.iloc[j][0])
        # 거리 계산
        dist = haversine(my_location, tour_place, unit='m')
        if int(dist) <= 2000:
            cnt += 1
    arr.append([cnt, my_location[0], my_location[1]])

savedf = pd.DataFrame(
    arr, columns=['tour_count', 'end_latitude', 'end_longitude'])
savedf.to_csv("AT4_2000.csv", index=False)
