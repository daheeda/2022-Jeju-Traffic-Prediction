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

    # weather = pd.read_csv('./jeju_data/jeju_weather.csv', encoding='cp949')
    # weather = weather[weather['지점명']=='제주']
    # weather['일시'] = make_Ymd(weather)
    # weather = weather.rename(columns={'일시': 'base_date'})
    # weather = weather.drop(['지점', '지점명'], axis=1)
    # print(weather)

    train_dist = make_dist(train)
    train['distance'] = train_dist
    train['week'] = make_week(train)
    train['week'] = train.apply(week_mapping, axis=1)
    # train['over_max_speed'] = over_max_speed(train) # Data leakage 의심
    train['time'] = group_time(train)
    cyclical_feature(train)
    train['month'] = make_month(train)
    train['post_holiday'] = make_post_holiday(post_holiday, train)
    train['pre_holiday'] = make_pre_holiday(pre_holiday, train)
    train['holiday'] = make_holiday2(train, holiday)
    train['season'] = group_season(train)
    train['vacation'] = vacation(train)
    # train['base_date'] = train['base_date'].astype('str')
    # train = pd.merge(train, holiday, on='base_date', how='left')
    # train['holiday'] = train['holiday'].fillna(0)
    # train['year'] = make_year(train)
    # train['day'] = make_day(train)
    # train['turn_road_rate'] = turn_road_rate(train)
    # train['end_turn_road_rate'] = end_turn_road_rate(train)
    # train['base_date'] = train['base_date'].astype('str')
    # train = pd.merge(train, weather, on='base_date', how='left')
    # train['일강수량(mm)'] = train['일강수량(mm)'].fillna(0)
    print('Train dataset success !')

    test_dist = make_dist(test)
    test['distance'] = test_dist
    test['week'] = make_week(test)
    test['week'] = test.apply(week_mapping, axis=1)
    # test['over_max_speed'] = over_max_speed(test) # Data leakage 의심
    test['time'] = group_time(test)
    cyclical_feature(test)
    test['month'] = make_month(test)
    test['post_holiday'] = make_post_holiday(post_holiday, test)
    test['pre_holiday'] = make_pre_holiday(pre_holiday, test)
    test['holiday'] = make_holiday2(test, holiday)
    test['season'] = group_season(test)
    test['vacation'] = vacation(test)
    # test['base_date'] = test['base_date'].astype('str')
    # test = pd.merge(test, holiday, on='base_date', how='left')
    # test['holiday'] = test['holiday'].fillna(0)
    # test['year'] = make_year(test)
    # test['day'] = make_day(test)
    # test['turn_road_rate'] = turn_road_rate(test)
    # test['end_turn_road_rate'] = end_turn_road_rate(test)
    # test['base_date'] = test['base_date'].astype('str')
    # test = pd.merge(test, weather, on='base_date', how='left')
    # test['일강수량(mm)'] = test['일강수량(mm)'].fillna(0)
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