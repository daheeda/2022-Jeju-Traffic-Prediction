import pandas as pd
from feature.feature_selection import *
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def make_dataset(train_path, test_path, holiday_path):    
    start = datetime.now()
    print('Start time: ', start)
    
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    test = pd.read_parquet(test_path)
    
    holiday = make_holiday(holiday_path)
    post_holiday = make_holiday(holiday_path)
    pre_holiday = make_holiday(holiday_path)

    cyclical_feature(train)
    train['group_time'] = group_time(train)
    train['month'] = make_month(train)
    train['week'] = make_week(train)
    train['week'] = train.apply(week_mapping, axis=1)
    train['post_holiday'] = make_post_holiday(post_holiday, train)
    train['pre_holiday'] = make_pre_holiday(pre_holiday, train)
    train['holiday'] = make_holiday2(train, holiday)
    train['season'] = group_season(train)
    train['vacation'] = vacation(train)
    train['distance'] = make_dist(train)
    train['road_name_new'] = road_name_new(train)
    train['weight_restricted_new'] = weight_restricted_new(train)
    train['day_of_week_2'] = day_of_week_2(train)
    train['base_hour_2'] = base_hour_2(train)
    print('Train dataset success !')

    cyclical_feature(test)
    test['group_time'] = group_time(test)
    test['month'] = make_month(test)
    test['week'] = make_week(test)
    test['week'] = test.apply(week_mapping, axis=1)
    test['post_holiday'] = make_post_holiday(post_holiday, test)
    test['pre_holiday'] = make_pre_holiday(pre_holiday, test)
    test['holiday'] = make_holiday2(test, holiday)
    test['season'] = group_season(test)
    test['vacation'] = vacation(test)
    test['distance'] = make_dist(test)
    test['road_name_new'] = road_name_new(test)
    test['weight_restricted_new'] = weight_restricted_new(test)
    test['day_of_week_2'] = day_of_week_2(test)
    test['base_hour_2'] = base_hour_2(test)
    print('Test dataset success !')

    train, test = make_cluster(train, test)

    rest_day(train)
    rest_day(test)

    move_lat_lng(train)
    move_lat_lng(test)

    interval(train)
    interval(test)

    lat_lng_scale(train)
    lat_lng_scale(test)

    lat_lng_minmax(train, test)

    train['turn_restricted'] = turn_restricted(train)
    test['turn_restricted'] = turn_restricted(test)


    # candidate = ['target']
    # for cand in candidate:  
    #     train = remove_outlier(train,cand)
    
    # train.reset_index(drop = True,inplace = True)
    
    train = train.drop(
    ['id', 'base_date',  'vehicle_restricted', 'post_date', 'pre_date', 'height_restricted'], axis=1
    )

    test = test.drop(
        ['id', 'base_date', 'vehicle_restricted', 'post_date', 'pre_date', 'height_restricted'], axis=1
    )

    End = datetime.now()
    print(f'End time: {End}' )
    print('Play time: ', End - start)
    
    return train, test