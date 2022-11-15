import pandas as pd
from haversine import haversine
from tqdm import tqdm
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim


def make_year(df):
    dt = df['base_date'].astype('str')
    month_data = pd.to_datetime(dt)
    md = month_data.dt.year
    return md


def make_month(df):
    dt = df['base_date'].astype('str')
    month_data = pd.to_datetime(dt)
    md = month_data.dt.month
    return md


def make_day(df):
    dt = df['base_date'].astype('str')
    month_data = pd.to_datetime(dt)
    md = month_data.dt.day
    return md


def group_season(df):
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2), 'season'] = '겨울'


    return df['season']


def turn_road_rate(df):
    df.loc[(df['start_turn_restricted'] == '있음') & (df['road_rating'] == 107), 'turn_road_rate'] = 0
    df.loc[(df['start_turn_restricted'] == '있음') & (df['road_rating'] == 103), 'turn_road_rate'] = 1
    df.loc[(df['start_turn_restricted'] == '없음') & (df['road_rating'] == 107), 'turn_road_rate'] = 2
    df.loc[(df['start_turn_restricted'] == '있음') & (df['road_rating'] == 106), 'turn_road_rate'] = 3
    df.loc[(df['start_turn_restricted'] == '없음') & (df['road_rating'] == 103), 'turn_road_rate'] = 4
    df.loc[(df['start_turn_restricted'] == '없음') & (df['road_rating'] == 106), 'turn_road_rate'] = 5
    return df['turn_road_rate']


def end_turn_road_rate(df):
    df.loc[(df['end_turn_restricted'] == '있음') & (df['road_rating'] == 107), 'end_turn_road_rate'] = 0
    df.loc[(df['end_turn_restricted'] == '있음') & (df['road_rating'] == 103), 'end_turn_road_rate'] = 1
    df.loc[(df['end_turn_restricted'] == '없음') & (df['road_rating'] == 107), 'end_turn_road_rate'] = 2
    df.loc[(df['end_turn_restricted'] == '있음') & (df['road_rating'] == 106), 'end_turn_road_rate'] = 3
    df.loc[(df['end_turn_restricted'] == '없음') & (df['road_rating'] == 103), 'end_turn_road_rate'] = 4
    df.loc[(df['end_turn_restricted'] == '없음') & (df['road_rating'] == 106), 'end_turn_road_rate'] = 5
    return df['end_turn_road_rate']


def make_dist(df):
    start_location = tuple(zip(df['start_latitude'], df['start_longitude']))
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(s, e) for s, e in zip(start_location, end_location)]

    return hsine


def make_week(df):
    dt = df['base_date'].astype('str')
    data = pd.to_datetime(dt)
    week = [i.weekday() for i in data]

    return week


def week_mapping(df):
    if df['week'] <= 4:
        val = 0
    else:
        val = 1
    return val


# cyclical continuous features - 24-hour time 주기성을 가지는 데이터를 알맞게 변환
def cyclical_feature(df):
    df['sin_time'] = np.sin(2*np.pi*df.base_hour/24)
    df['cos_time'] = np.cos(2*np.pi*df.base_hour/24)


def cyclical_feature2(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    

def vacation(df):
    df.loc[(df['month'] == 7) | (df['month'] == 8) | (df['month'] == 1) | (df['month'] == 2), 'vacation'] = 'vacation'
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5) | (df['month'] == 6) | (df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11) | (df['month'] == 12), 'vacation'] = 'semester'

    return df['vacation']

    
def over_max_speed(df):
    df.loc[(df['maximum_speed_limit'] == 30), 'over_max_speed'] = 1
    df.loc[(df['maximum_speed_limit'] == 40), 'over_max_speed'] = 1
    df.loc[(df['maximum_speed_limit'] == 50), 'over_max_speed'] = 0
    df.loc[(df['maximum_speed_limit'] == 60), 'over_max_speed'] = 0
    df.loc[(df['maximum_speed_limit'] == 70), 'over_max_speed'] = 0
    df.loc[(df['maximum_speed_limit'] == 80), 'over_max_speed'] = 0
    
    return df['over_max_speed']


def make_Ymd(df):
    dt = df['일시'].astype('str')
    month_data = pd.to_datetime(dt).dt.strftime("%Y%m%d")
    return month_data


def geocoding_reverse(lat_lng_str): 
    geolocoder = Nominatim(user_agent = 'South Korea', timeout=None)
    address = geolocoder.reverse(lat_lng_str)

    return address


def weather(df):
    df['base_date'] = df['base_date'].astype('str')
    merge = pd.merge(df, weather, on='base_date', how='left')
    df['일강수량(mm)'] = merge['일강수량(mm)'].fillna(0)


def group_time(df):
    df.loc[(df['base_hour'] < 6), 'time'] = '새벽'
    df.loc[(df['base_hour'] >=6) & (df['base_hour'] < 12), 'time'] = '아침'
    df.loc[(df['base_hour'] >= 12) & (df['base_hour'] < 19), 'time'] = '오후'
    df.loc[(df['base_hour'] >= 19) & (df['base_hour'] <= 24), 'time'] = '저녁'

    return df['time']


def make_holiday(path):
    holiday = pd.read_csv(path)
    holiday['Year'] = holiday['Year'].astype('str')
    holiday['Month'] = holiday['Month'].astype('str')
    holiday['Day'] = holiday['Day'].astype('str')

    re_month = [holiday['Month'][i].zfill(2) for i in range(len(holiday))]
    re_day = [holiday['Day'][i].zfill(2) for i in range(len(holiday))]

    holiday['Month'] = re_month
    holiday['Day'] = re_day
    holiday['base_date'] = holiday['Year'] + holiday['Month'] + holiday['Day']
    holiday['holiday'] = 1

    holiday = holiday.drop(['Year', 'Month', 'Day', 'Info'], axis=1)

    return holiday


def make_holiday2(df, holiday):
    df['base_date'] = df['base_date'].astype('str')
    df = pd.merge(df, holiday, on='base_date', how='left')
    df['holiday'] = df['holiday'].fillna(0)

    return df['holiday']


def make_post_holiday(holiday, df):
    holiday_date = holiday['base_date']
    holiday_date = pd.to_datetime(holiday_date)
    post_holiday = holiday_date - pd.Timedelta(days=1)
    holiday['post_date'] = post_holiday
    holiday = holiday.drop(['base_date'], axis=1)
    holiday = holiday.rename(columns={'holiday': 'post_holiday'})
    
    df['post_date'] = df['base_date']
    df['post_date'] = df['post_date'].astype('str')
    df['post_date']  = pd.to_datetime(df['post_date'] )
    
    df_merge_p = pd.merge(df, holiday, on='post_date', how='left')
    df_merge_p['post_holiday'] = df_merge_p['post_holiday'].fillna(0)
    
    return df_merge_p['post_holiday']


def make_pre_holiday(holiday, df):
    holiday_date = holiday['base_date']
    holiday_date = pd.to_datetime(holiday_date)
    pre_holiday = holiday_date + pd.Timedelta(days=1)
    holiday['pre_date'] = pre_holiday
    holiday = holiday.drop(['base_date'], axis=1)
    holiday = holiday.rename(columns={'holiday': 'pre_holiday'})
    
    df['pre_date'] = df['base_date']
    df['pre_date'] = df['pre_date'].astype('str')
    df['pre_date']  = pd.to_datetime(df['pre_date'] )
    
    df_merge = pd.merge(df, holiday, on='pre_date', how='left')
    df_merge['pre_holiday'] = df_merge['pre_holiday'].fillna(0)
    
    return df_merge['pre_holiday']


def make_cluster(train, test):
    from sklearn.cluster import KMeans
    train_c = train[['start_latitude', 'start_longitude']]
    test_c = test[['start_latitude', 'start_longitude']]
    
    k_mean = KMeans(n_clusters=4, init='k-means++')
    train['location_cluster'] = k_mean.fit_predict(train_c)
    test['location_cluster'] = k_mean.predict(test_c)
    
    return train, test    



def remove_outlier(train, column):
    import numpy as np
    df = train[column]
    
    # 1분위수
    quan_25 = np.percentile(df.values, 25)
    
    # 3분위수
    quan_75 = np.percentile(df.values, 75)
    
    iqr = quan_75 - quan_25
    
    lowest = quan_25 - iqr * 1.5
    highest = quan_75 + iqr * 1.5
    outlier_index = df[(df < lowest) | (df > highest)].index
    print('outlier의 수 : ' , len(outlier_index))
    train.drop(outlier_index, axis = 0, inplace = True)
    
    return train


def rest_day(X):
    X['week'] = X['week'].astype('float')
    X['new'] = X['week'] + X['pre_holiday'] + X['holiday'] + X['post_holiday']
    X.loc[(X['new'] >= 1), 'rest'] = 1
    X.loc[(X['new'] == 0), 'rest'] = 0
    
    
    
'''
def make_post_holiday(holiday, df):
    holiday_date = holiday['base_date']
    holiday_date = pd.to_datetime(holiday_date)
    post_holiday = holiday_date + pd.Timedelta(days=1)
    holiday['post_date'] = post_holiday
    holiday = holiday.drop(['base_date'], axis=1)
    holiday = holiday.rename(columns={'holiday': 'post_holiday'})
    
    df_date = df['base_date'].astype('str')
    df_date = pd.to_datetime(df_date)
    post_df = df_date + pd.Timedelta(days=1)
    df['post_date'] = post_df
    
    df_merge_p = pd.merge(df, holiday, on='post_date', how='left')
    df_merge_p['post_holiday'] = df_merge_p['post_holiday'].fillna(0)
    
    return df_merge_p['post_holiday']


def make_pre_holiday(holiday, df):
    holiday_date = holiday['base_date']
    holiday_date = pd.to_datetime(holiday_date)
    pre_holiday = holiday_date - pd.Timedelta(days=1)
    holiday['pre_date'] = pre_holiday
    holiday = holiday.drop(['base_date'], axis=1)
    holiday = holiday.rename(columns={'holiday': 'pre_holiday'})
    
    df_date = df['base_date'].astype('str')
    df_date = pd.to_datetime(df_date)
    pre_df = df_date - pd.Timedelta(days=1)
    df['pre_date'] = pre_df
    
    df_merge = pd.merge(df, holiday, on='pre_date', how='left')
    df_merge['pre_holiday'] = df_merge['pre_holiday'].fillna(0)
    
    return df_merge['pre_holiday']
'''