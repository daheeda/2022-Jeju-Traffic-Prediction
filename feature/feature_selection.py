import pandas as pd
import numpy as np
from haversine import haversine
from sklearn.cluster import KMeans


def cyclical_feature(df): # V if string 
    df['sin_time'] = np.sin(2*np.pi*df.base_hour/24)
    df['cos_time'] = np.cos(2*np.pi*df.base_hour/24)

      
def group_time(df): # O
    df['group_time'] = '-' 
    df.loc[(df['base_hour'] < 6), 'group_time'] = '새벽'
    df.loc[(df['base_hour'] >=6) & (df['base_hour'] < 12), 'group_time'] = '아침'
    df.loc[(df['base_hour'] >= 12) & (df['base_hour'] < 19), 'group_time'] = '오후'
    df.loc[(df['base_hour'] >= 19) & (df['base_hour'] <= 24), 'group_time'] = '저녁'
    df.loc[(df['group_time']=='-'), 'group_time'] = 'Na'
    return df['group_time']


def make_month(df): # O
    dt = df['base_date'].astype('str')
    month_data = pd.to_datetime(dt)
    md = month_data.dt.month
    return md


def group_season(df): # O
    df['season'] = '-'
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2), 'season'] = '겨울'
    df.loc[(df['season']=='-'), 'season'] = 'Na'
    return df['season']


def make_week(df): # V if null
    dt = df['base_date'].astype('str')
    data = pd.to_datetime(dt)
    week = [i.weekday() for i in data]
    df['week'] = week
    df.loc[(df['week'] <= 4), 'week'] = 0
    df.loc[(df['week'] > 4), 'week'] = 1
    return df['week']


def vacation(df): # O
    df['vacation'] = '-'
    df.loc[(df['month'] == 7) | (df['month'] == 8) | (df['month'] == 1) | (df['month'] == 2), 'vacation'] = 'vacation'
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5) | (df['month'] == 6) | (df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11) | (df['month'] == 12), 'vacation'] = 'semester'
    df.loc[(df['vacation']=='-'), 'vacation'] = 'Na'
    return df['vacation']


def make_holiday(path): # O
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


def make_holiday2(df, holiday): # V
    df['base_date'] = df['base_date'].astype('str')
    df = pd.merge(df, holiday, on='base_date', how='left')
    df['holiday'] = df['holiday'].fillna(0)

    return df['holiday']


def make_post_holiday(holiday, df): # V
    holiday_date = holiday['base_date']
    holiday_date = pd.to_datetime(holiday_date)
    post_holiday = holiday_date - pd.Timedelta(days=1)
    holiday['post_date'] = post_holiday
    holiday = holiday.drop(['base_date'], axis=1)
    holiday = holiday.rename(columns={'holiday': 'post_holiday'})
    
    df['post_date'] = df['base_date']
    df['post_date'] = df['post_date'].astype('str')
    df['post_date'] = pd.to_datetime(df['post_date'] )
    
    df_merge_p = pd.merge(df, holiday, on='post_date', how='left')
    df_merge_p['post_holiday'] = df_merge_p['post_holiday'].fillna(0)
    
    return df_merge_p['post_holiday']


def make_pre_holiday(holiday, df): # V if null
    holiday_date = holiday['base_date']
    holiday_date = pd.to_datetime(holiday_date)
    pre_holiday = holiday_date + pd.Timedelta(days=1)
    holiday['pre_date'] = pre_holiday
    holiday = holiday.drop(['base_date'], axis=1)
    holiday = holiday.rename(columns={'holiday': 'pre_holiday'})
    
    df['pre_date'] = df['base_date']
    df['pre_date'] = df['pre_date'].astype('str')
    df['pre_date'] = pd.to_datetime(df['pre_date'] )
    
    df_merge = pd.merge(df, holiday, on='pre_date', how='left')
    df_merge['pre_holiday'] = df_merge['pre_holiday'].fillna(0)
    
    return df_merge['pre_holiday']


def rest_day(df): # O
    df['week'] = df['week'].astype('float')
    df['rest'] = df['week'] + df['pre_holiday'] + df['holiday'] + df['post_holiday']
    df.loc[(df['rest'] >= 1), 'rest'] = 1
    df.loc[(df['rest'] == 0), 'rest'] = 0
    
    
def make_dist(df): # V if null 
    start_location = tuple(zip(df['start_latitude'], df['start_longitude']))
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(s, e) for s, e in zip(start_location, end_location)]

    return hsine


def make_cluster(train, test): # O
    train_c = train[['start_latitude', 'start_longitude']]
    test_c = test[['start_latitude', 'start_longitude']]
    cluster_centers = np.array([[33.26345514655621116162365069612860679626464843, 126.5203815031463392415389535017311573028564453],
                                [33.37082277149481512878992361947894096374511718, 126.2976713570606790426609222777187824249267578],
                                [33.48077890914120757770433556288480758666992187, 126.4946717292079512162672472186386585235595703],
                                [33.41815597422977646147046471014618873596191406, 126.7739831436176700663054361939430236816406250]])

    k_mean = KMeans(n_clusters=4, init=cluster_centers , random_state = 2)
    train['location_cluster'] = k_mean.fit_predict(train_c)
    test['location_cluster'] = k_mean.predict(test_c)
    
    return train, test    

# O
def jeju_dist(df):
    jeju_location = (33.4996213, 126.5311884)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]
    return hsine

def seogwi_dist(df):
    jeju_location = (33.2541205, 126.560076)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine

def hanra_dist(df):
    jeju_location = (33.361417, 126.529417)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine


def sungsan_dist(df):
    jeju_location = (33.458528, 126.94225)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine


def joongmoon_dist(df):
    jeju_location = (33.246340915095914, 126.41973291093717)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine


def turn_restricted(df): # O
    df['turn_restricted'] = df['start_turn_restricted'] + df['end_turn_restricted']

    return df['turn_restricted']


def speed(train, test, col, col_name): # O
    speed = train.groupby([col, 'maximum_speed_limit'])['target'].agg([(col_name, 'mean')]).reset_index()
    train = pd.merge(train, speed, on=[col, 'maximum_speed_limit'], how='left')
    test = pd.merge(test, speed, on=[col, 'maximum_speed_limit'], how='left')
    test[col_name] = test[col_name].fillna(train[col_name].mode())
    return train, test


def speed_time(train, test, col, col_name): # O
    speed = train.groupby([col, 'base_hour'])['target'].agg([(col_name, 'mean')]).reset_index()
    train = pd.merge(train, speed, on=[col, 'base_hour'], how='left')
    test = pd.merge(test, speed, on=[col, 'base_hour'], how='left')
    test[col_name] = test[col_name].fillna(train[col_name].mode())
    return train, test


def node_tf(train, test): # O
    train['node_TF'] = train['start_node_name'] == train['end_node_name']
    test['node_TF'] = test['start_node_name'] == test['end_node_name']
    return train, test
    
    
def sm_tm(train, test): # O st_mean = speed target mean
    st_mean = train.groupby('maximum_speed_limit')['target'].agg([('sm_tm', 'mean')]).reset_index()
    st_mean['diff'] = st_mean['maximum_speed_limit'] - st_mean['sm_tm']
    st_mean = st_mean.drop(['sm_tm'], axis=1)
    train = pd.merge(train, st_mean, on=['maximum_speed_limit'], how='left')
    test = pd.merge(test, st_mean, on=['maximum_speed_limit'], how='left')
    test['diff'] = test['diff'].fillna(train['diff'].mode())
    return train, test


def road_name_set(train, test): # O
    train.loc[train['road_name'][(train['road_type'] == 3)].index, 'road_name'] = '국_지_도'
    test.loc[test['road_name'][(test['road_type'] == 3)].index, 'road_name'] = '국_지_도'

    train['road_name_set'] = '0'
    train.loc[train['road_name'].str.contains('국도'), 'road_name_set'] = 'A'
    train.loc[train['road_name'].str.contains('지방도'), 'road_name_set'] = 'A'
    train.loc[train['road_name'].str.contains('국_지_도'), 'road_name_set'] = 'A'
    train.loc[train['road_name'].str.contains('로'), 'road_name_set'] = 'B'
    train.loc[train['road_name'].str.contains('교'), 'road_name_set'] = 'C'
    

    test['road_name_set'] = '0'
    test.loc[test['road_name'].str.contains('국도'), 'road_name_set'] = 'A'
    test.loc[test['road_name'].str.contains('지방도'), 'road_name_set'] = 'A'
    test.loc[test['road_name'].str.contains('국_지_도'), 'road_name_set'] = 'A'
    test.loc[test['road_name'].str.contains('로'), 'road_name_set'] = 'B'
    test.loc[test['road_name'].str.contains('교'), 'road_name_set'] = 'C'
    
    return train, test


def Tourist(df, tour_df): # O
    tour_df['end_cartesian'] = tour_df['end_latitude'].astype('str') + ',' + tour_df['end_longitude'].astype('str')
    df['end_cartesian'] = df['end_latitude'].astype('str') + ',' + df['end_longitude'].astype('str')
    tour_df = tour_df.drop(['end_latitude', 'end_longitude'], axis=1)
    df = pd.merge(df, tour_df, how='left', on='end_cartesian')
    df['tour_count'] = df['tour_count'].fillna(tour_df['tour_count'].mode())
    return df
