import pandas as pd
from feature.feature_selection import *
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def make_dataset(train_path, test_path, holiday_path):
    
    start = datetime.now()
    print('Start time: ', start)
    
    train = pd.read_parquet(train_path)
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
    
    print('Test dataset success !')

    str_col = ['day_of_week', 'start_turn_restricted', 'end_turn_restricted', 'group_time', 'season', 'vacation']
    
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
    
    # candidate = ['target']
    # for cand in candidate:  
    #     train = remove_outlier(train,cand)
    
    # train.reset_index(drop = True,inplace = True)
    
    X = train.drop(
    ['id', 'base_date', 'target', 'road_name', 'start_node_name', 'end_node_name', 'vehicle_restricted', 'base_hour', 'post_date', 'pre_date'], axis=1
    )

    y = train['target']

    test = test.drop(
        ['id', 'base_date', 'road_name', 'start_node_name', 'end_node_name', 'vehicle_restricted', 'base_hour', 'post_date', 'pre_date'], axis=1
    )

    End = datetime.now()
    print(f'End time: {End}' )
    print('Play time: ', End - start)
    
    return X, y, test