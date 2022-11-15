from datetime import datetime

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from feature.feature_selection import *


def make_dataset(train_path, test_path, holiday_path, tour_path):
    
    start = datetime.now()
    print('Start time: ', start)
    
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    tour_list = pd.read_csv(tour_path)
    
    holiday = make_holiday(holiday_path)
    post_holiday = make_holiday(holiday_path)
    pre_holiday = make_holiday(holiday_path)
    
    ### 이 시점 ###    
                
    cyclical_feature(train)
    train['group_time'] = group_time(train)
    
    train['month'] = make_month(train)
    train['week'] = make_week(train)
    train['post_holiday'] = make_post_holiday(post_holiday, train)
    train['pre_holiday'] = make_pre_holiday(pre_holiday, train)
    train['holiday'] = make_holiday2(train, holiday)
    train['season'] = group_season(train)
    train['vacation'] = vacation(train)
    
    train['distance'] = make_dist(train)
    train['jeju_dist'] = jeju_dist(train)
    train['seogwi_dist'] = seogwi_dist(train)
    train['hanra_dist'] = hanra_dist(train)
    train['sungsan_dist'] = sungsan_dist(train)
    train['joongmoon_dist'] = joongmoon_dist(train)
    
    print('Train dataset success !')

    cyclical_feature(test)
    test['group_time'] = group_time(test)
    
    test['month'] = make_month(test)
    test['week'] = make_week(test)
    test['post_holiday'] = make_post_holiday(post_holiday, test)
    test['pre_holiday'] = make_pre_holiday(pre_holiday, test)
    test['holiday'] = make_holiday2(test, holiday)
    test['season'] = group_season(test)
    test['vacation'] = vacation(test)
    
    test['distance'] = make_dist(test)
    test['jeju_dist'] = jeju_dist(test)
    test['seogwi_dist'] = seogwi_dist(test)
    test['hanra_dist'] = hanra_dist(test)
    test['sungsan_dist'] = sungsan_dist(test)
    test['joongmoon_dist'] = joongmoon_dist(test)
    
    print('Test dataset success !')

    train, test = node_tf(train, test)
    train, test = sm_tm(train, test)
    train, test =road_name_set(train, test)
    
    train, test = speed_time(train,test,'road_name','section_speed_time')
    train, test = speed_time(train,test,'start_node_name','start_speed_time')
    train, test = speed_time(train,test,'end_node_name','end_speed_time')
    
    train, test = speed(train,test,'road_name','section_speed')
    train, test = speed(train,test,'start_node_name','start_speed')
    train, test = speed(train,test,'end_node_name','end_speed')
    
    train = Tourist(train, tour_list)
    test = Tourist(test, tour_list)
    
    train["node_TF"] = train["node_TF"].astype(int)
    test["node_TF"] = test["node_TF"].astype(int)
    
    
    str_col = ['day_of_week', 'start_turn_restricted', 'end_turn_restricted',
               'road_name', 'start_node_name', 'end_node_name', 'group_time',
               'season', 'vacation', 'road_name_set', 'end_cartesian']
    
    for i in str_col:
        le = LabelEncoder()
        le = le.fit(train[i])
        train[i] = le.transform(train[i])

        for label in np.unique(test[i]):
            if label not in le.classes_:
                le.classes_ = np.append(le.classes_, label)
        test[i] = le.transform(test[i])

    train['turn_restricted'] = turn_restricted(train)
    test['turn_restricted'] = turn_restricted(test)

    rest_day(train)
    rest_day(test)
    
    train, test = make_cluster(train, test)
    
    X = train.drop(    
        ['id', 'base_date', 'target', 'vehicle_restricted', 'height_restricted',
        'post_date', 'pre_date'], axis=1
    )

    y = train['target']

    test = test.drop(
        ['id', 'base_date', 'vehicle_restricted', 'height_restricted',
         'post_date', 'pre_date'], axis=1
    )

    End = datetime.now()
    print(f'End time: {End}')
    print('Play time: ', End - start)
    
    return X, y, test