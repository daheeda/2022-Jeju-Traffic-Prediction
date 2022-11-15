import pandas as pd
from feature_selection import *
from sklearn.preprocessing import LabelEncoder
from datetime import date, datetime


def make_dataset(train_path, test_path):
    
    start = datetime.now()
    print('Start time: ', start)
    
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    holiday = make_holiday('./jeju_data/국가공휴일.csv')
    post_holiday = make_holiday('./jeju_data/국가공휴일.csv')
    pre_holiday = make_holiday('./jeju_data/국가공휴일.csv')

    train_dist = make_dist(train)
    train['distance'] = train_dist
    train['week'] = make_week(train)
    train['week'] = train.apply(week_mapping, axis=1)
    train['time'] = group_time(train)
    cyclical_feature(train)
    train['month'] = make_month(train)
    train['post_holiday'] = make_post_holiday(post_holiday, train)
    train['pre_holiday'] = make_pre_holiday(pre_holiday, train)
    train['holiday'] = make_holiday2(train, holiday)
    train['season'] = group_season(train)
    train['vacation'] = vacation(train)
    
    print('Train dataset success !')

    test_dist = make_dist(test)
    test['distance'] = test_dist
    test['week'] = make_week(test)
    test['week'] = test.apply(week_mapping, axis=1)
    test['time'] = group_time(test)
    cyclical_feature(test)
    test['month'] = make_month(test)
    test['post_holiday'] = make_post_holiday(post_holiday, test)
    test['pre_holiday'] = make_pre_holiday(pre_holiday, test)
    test['holiday'] = make_holiday2(test, holiday)
    test['season'] = group_season(test)
    test['vacation'] = vacation(test)

    print('Test dataset success !')

    str_col = ['day_of_week', 'start_turn_restricted', 'end_turn_restricted', 'time', 'season', 'vacation']
    
    for i in str_col:
        le = LabelEncoder()
        le = le.fit(train[i])
        train[i] = le.transform(train[i])

        for label in np.unique(test[i]):
            if label not in le.classes_:
                le.classes_ = np.append(le.classes_, label)
        test[i] = le.transform(test[i])

    train['turn_restricted'] = train['start_turn_restricted'] + train['end_turn_restricted']
    test['turn_restricted'] = test['start_turn_restricted'] + test['end_turn_restricted']
    
    train, test = make_cluster(train, test)
    # candidate = ['target']
    # for cand in candidate:  
    #     train = remove_outlier(train,cand)

    rest_day(train)
    rest_day(test)
    
    train.reset_index(drop = True,inplace = True)
    
    X = train.drop(
    ['id', 'base_date', 'target', 'road_name', 'start_node_name', 'end_node_name', 'vehicle_restricted', 'base_hour', 'post_date', 'pre_date', 'new'], axis=1
    )

    y = train['target']

    test = test.drop(
        ['id', 'base_date', 'road_name', 'start_node_name', 'end_node_name', 'vehicle_restricted', 'base_hour', 'post_date', 'pre_date', 'new'], axis=1
    )
        
    
    
    
    End = datetime.now()
    print(f'End time: {End}' )
    print('Play time: ', End - start)
    
    return X, y, test