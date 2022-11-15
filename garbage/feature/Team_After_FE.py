import pandas as pd
from haversine import haversine
from tqdm import tqdm
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.preprocessing import OneHotEncoder


'''
Feature selection

시간 (0 ~ 24)
1. cyclical_feature: 24시간을 주기성을 가지는 데이터로 변환
2. group_time: 새벽, 아침, 점심, 저녁


요일(week)
1. make_week: 공휴일
2. make_holiday: 주말
3. make_post_holiday, make_pre_holiday: 전날이 공휴일, 다음날이 공휴일


날짜(Ymd)
1. make_month: 달
2. group_season: 봄, 여름, 가을, 겨율

위도, 경도
1. make_dist: 두 지점 사이의 거리
2. make_cluster: 지역변수로 clustering => 4가지로 분류
    => 제주시, 서귀포시 주변으로 분류됨

'''

######################### 시간 #############################


def cyclical_feature(df):
    df['sin_time'] = np.sin(2*np.pi*df.base_hour/24)
    df['cos_time'] = np.cos(2*np.pi*df.base_hour/24)


# def cyclical_feature2(df, col, max_val):
#     df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
#     df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
      
'''      
def group_time(df): 
    df.loc[(df['base_hour'] < 6), 'group_time'] = '새벽'
    df.loc[(df['base_hour'] >=6) & (df['base_hour'] < 12), 'group_time'] = '아침'
    df.loc[(df['base_hour'] >= 12) & (df['base_hour'] < 19), 'group_time'] = '오후'
    df.loc[(df['base_hour'] >= 19) & (df['base_hour'] <= 24), 'group_time'] = '저녁'

    return df['group_time']
#라벨을 원핫으로 바꿔줄거임 밑 함
'''
def group_time(train_x, test_x): 
    train_x['lazy_hour'] = 0
    train_x['lazy_hour'][(train_x['base_hour'] == 0)] = 'c'
    train_x['lazy_hour'][(train_x['base_hour'] == 1)] = 'c'
    train_x['lazy_hour'][(train_x['base_hour'] == 2)] = 'c'
    train_x['lazy_hour'][(train_x['base_hour'] == 3)] = 'c'
    train_x['lazy_hour'][(train_x['base_hour'] == 4)] = 'c'
    train_x['lazy_hour'][(train_x['base_hour'] == 5)] = 'a'
    train_x['lazy_hour'][(train_x['base_hour'] == 6)] = 'a'
    train_x['lazy_hour'][(train_x['base_hour'] == 7)] = 'a'
    train_x['lazy_hour'][(train_x['base_hour'] == 8)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 9)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 10)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 11)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 12)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 13)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 14)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 15)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 16)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 17)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 18)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 19)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 20)] = 'b'
    train_x['lazy_hour'][(train_x['base_hour'] == 21)] = 'a'
    train_x['lazy_hour'][(train_x['base_hour'] == 22)] = 'a'
    train_x['lazy_hour'][(train_x['base_hour'] == 23)] = 'a'

    test_x['lazy_hour'] = 0 #Test의 base_hour에 0~23이외의 값이 존재할 수 있으니 일단 0으로 : DataLeakage 방지용
    test_x['lazy_hour'][(test_x['base_hour'] == 0)] = 'c'
    test_x['lazy_hour'][(test_x['base_hour'] == 1)] = 'c'
    test_x['lazy_hour'][(test_x['base_hour'] == 2)] = 'c'
    test_x['lazy_hour'][(test_x['base_hour'] == 3)] = 'c'
    test_x['lazy_hour'][(test_x['base_hour'] == 4)] = 'c'
    test_x['lazy_hour'][(test_x['base_hour'] == 5)] = 'a'
    test_x['lazy_hour'][(test_x['base_hour'] == 6)] = 'a'
    test_x['lazy_hour'][(test_x['base_hour'] == 7)] = 'a'
    test_x['lazy_hour'][(test_x['base_hour'] == 8)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 9)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 10)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 11)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 12)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 13)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 14)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 15)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 16)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 17)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 18)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 19)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 20)] = 'b'
    test_x['lazy_hour'][(test_x['base_hour'] == 21)] = 'a'
    test_x['lazy_hour'][(test_x['base_hour'] == 22)] = 'a'
    test_x['lazy_hour'][(test_x['base_hour'] == 23)] = 'a'

    ohe = OneHotEncoder(sparse=False)
    
    train_cat = ohe.fit_transform(train_x[['lazy_hour']])
    pd.DataFrame(train_cat, columns=['lazy_hour' + col for col in ohe.categories_[0]])
    train_x = pd.concat([train_x.drop(columns=['lazy_hour']), pd.DataFrame(train_cat, columns=['lazy_hour' + col for col in ohe.categories_[0]])], axis=1)

    test_cat = ohe.transform(test_x[['lazy_hour']])
    pd.DataFrame(test_cat, columns=['lazy_hour' + col for col in ohe.categories_[0]])
    test_x = pd.concat([test_x.drop(columns=['lazy_hour']), pd.DataFrame(test_cat, columns=['lazy_hour' + col for col in ohe.categories_[0]])], axis=1)

    return train_x, test_x

######################### 날짜 #############################
'''
def make_Ymd(df):
    dt = df['일시'].astype('str')
    month_data = pd.to_datetime(dt).dt.strftime("%Y%m%d")
    return month_data


def make_year(df):
    dt = df['base_date'].astype('str')
    month_data = pd.to_datetime(dt)
    md = month_data.dt.year
    return md
'''

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

'''
def group_season(df):
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2), 'season'] = '겨울'

    return df['season']
#라벨을 원핫으로 바꿔줄거임 밑 함수
'''

def group_season(train_x, test_x):
    train_x['month'] = (train_x["base_date"] % 10000) // 100
    train_x['season'] = 'a'
    train_x['season'][(train_x['month'] == 7)] = 'b'
    train_x['season'][(train_x['month'] == 8)] = 'b'

    train_x.drop(['month'], axis = 1 , inplace = True)
    train_x['month_day'] = (train_x["base_date"] % 10000) #공휴일
    #train_x.drop(['base_date'], axis = 1 , inplace = True)

    test_x['month'] = (test_x["base_date"] % 10000) // 100 
    test_x['season'] = 'a'
    test_x['season'][(test_x['month'] == 8)] = 'b'
    test_x['season'][(test_x['month'] == 8)] = 'b'

    test_x.drop(['month'], axis = 1 , inplace = True)
    test_x['month_day'] = (test_x["base_date"] % 10000) #공휴일
    #test_x.drop(['base_date'], axis = 1 , inplace = True)

    ohe = OneHotEncoder(sparse=False)
    
    train_cat = ohe.fit_transform(train_x[['season']])
    pd.DataFrame(train_cat, columns=['season' + col for col in ohe.categories_[0]])
    train_x = pd.concat([train_x.drop(columns=['season']), pd.DataFrame(train_cat, columns=['season' + col for col in ohe.categories_[0]])], axis=1)

    test_cat = ohe.transform(test_x[['season']])
    pd.DataFrame(test_cat, columns=['season' + col for col in ohe.categories_[0]])
    test_x = pd.concat([test_x.drop(columns=['season']), pd.DataFrame(test_cat, columns=['season' + col for col in ohe.categories_[0]])], axis=1)

    return train_x, test_x

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

'''
def vacation(df):
    df.loc[(df['month'] == 7) | (df['month'] == 8) | (df['month'] == 1) | (df['month'] == 2), 'vacation'] = 'vacation'
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5) | (df['month'] == 6) | (df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11) | (df['month'] == 12), 'vacation'] = 'semester'

    return df['vacation']
'''

    
######################### 위도 경도 #############################

def make_dist(df):
    start_location = tuple(zip(df['start_latitude'], df['start_longitude']))
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(s, e) for s, e in zip(start_location, end_location)]

    return hsine

'''
def geocoding_reverse(lat_lng_str): 
    geolocoder = Nominatim(user_agent='South Korea', timeout=None)
    address = geolocoder.reverse(lat_lng_str)

    return address
'''

def make_cluster(train, test):
    from sklearn.cluster import KMeans
    train_c = train[['start_latitude', 'start_longitude']]
    test_c = test[['start_latitude', 'start_longitude']]
    
    k_mean = KMeans(n_clusters=4, init='k-means++')
    train['location_cluster'] = k_mean.fit_predict(train_c)
    test['location_cluster'] = k_mean.predict(test_c)
    
    return train, test    

############### 기타 ##############################


def turn_restricted(df):
    df['turn_restricted'] = df['start_turn_restricted'] + df['end_turn_restricted']

    return df['turn_restricted']


def over_max_speed(train_x,test_x):
    train_x['maximum_speed_limit_E'] = 'a'
    train_x['maximum_speed_limit_E'][(train_x['maximum_speed_limit'] == 30)] = 'b'
    train_x['maximum_speed_limit_E'][(train_x['maximum_speed_limit'] == 40)] = 'b'
    train_x['maximum_speed_limit_E'][(train_x['maximum_speed_limit'] == 50)] = 'b'
    train_x['maximum_speed_limit_E'][(train_x['maximum_speed_limit'] == 60)] = 'a'
    train_x['maximum_speed_limit_E'][(train_x['maximum_speed_limit'] == 70)] = 'a'
    train_x['maximum_speed_limit_E'][(train_x['maximum_speed_limit'] == 80)] = 'a'

    test_x['maximum_speed_limit_E'] = 'a'
    test_x['maximum_speed_limit_E'][(test_x['maximum_speed_limit'] == 30)] = 'b'
    test_x['maximum_speed_limit_E'][(test_x['maximum_speed_limit'] == 40)] = 'b'
    test_x['maximum_speed_limit_E'][(test_x['maximum_speed_limit'] == 50)] = 'b'
    test_x['maximum_speed_limit_E'][(test_x['maximum_speed_limit'] == 60)] = 'a'
    test_x['maximum_speed_limit_E'][(test_x['maximum_speed_limit'] == 70)] = 'a'
    test_x['maximum_speed_limit_E'][(test_x['maximum_speed_limit'] == 80)] = 'a'
    ohe = OneHotEncoder(sparse=False)

    train_cat = ohe.fit_transform(train_x[['maximum_speed_limit_E']])
    pd.DataFrame(train_cat, columns=['maximum_speed_limit_E' + col for col in ohe.categories_[0]])
    train_x = pd.concat([train_x.drop(['maximum_speed_limit_E'], axis = 1), pd.DataFrame(train_cat, columns=['maximum_speed_limit_E' + col for col in ohe.categories_[0]])], axis=1)


    # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
    test_cat = ohe.transform(test_x[['maximum_speed_limit_E']])
    pd.DataFrame(test_cat, columns=['maximum_speed_limit_E' + col for col in ohe.categories_[0]])
    test_x = pd.concat([test_x.drop(['maximum_speed_limit_E'], axis = 1), pd.DataFrame(test_cat, columns=['maximum_speed_limit_E' + col for col in ohe.categories_[0]])], axis=1)

    return train_x, test_x
    
###########################################################
####################### 팀원 ################################
###########################################################

def move_lat_lng(df):
      df['move_lat'] = abs(df['start_latitude']-df['end_latitude'])*1000
      df['move_lng'] = abs(df['start_longitude']-df['end_longitude'])*1000
      
      
def interval(df):
    df['start_cartesian'] = df['start_latitude'].astype('str') + ',' + df['start_longitude'].astype('str')
    df['end_cartesian'] = df['end_latitude'].astype('str') + ',' + df['end_longitude'].astype('str')
    #df['end_cartesian'] = df['end_cartesian'] + 32
    #df['start_cartesian'] = df['start_cartesian'] + 32
    #df['interval'] = df['start_node_name'] + ',' + df['end_node_name']


def lat_lng_scale(df):
    df['start_latitude'] = df['start_latitude'] - 33
    df['end_latitude'] = df['start_latitude'] - 33
    df['start_longitude'] = df['start_latitude'] - 126
    df['end_longitude'] = df['start_latitude'] - 126


def lat_lng_minmax(train, test):
    from sklearn.preprocessing import MinMaxScaler

    location_col = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']
    scaler = MinMaxScaler()
    for i in location_col:
        train[i] = scaler.fit_transform(train[i].values.reshape(-1, 1))
        test[i] = scaler.transform(test[i].values.reshape(-1, 1))


def road_name_new(train_x,test_x):  # NaN 값 들어가있는지 확인필요
    train_x['road_name'][(train_x['road_type'] == 3)] = '국_지_도'
    test_x['road_name'][(test_x['road_type'] == 3)] = '국_지_도'
    train_x['rname_new'] = '0'
    train_x['rname_new'][train_x['road_name'].str.contains('국도')] = 'a'
    train_x['rname_new'][train_x['road_name'].str.contains('지방도')] = 'a'
    train_x['rname_new'][train_x['road_name'].str.contains('로')] = 'b'
    train_x['rname_new'][train_x['road_name'].str.contains('교')] = 'c'
    train_x['rname_new'][train_x['road_name'].str.contains('국_지_도')] = 'a'
    #train_x['rname_new'][train_x['road_name'].str.contains('NaN')] = 'b'
    train_x['rname_new'][train_x['road_name'] == '0'] = 'a'


    test_x['rname_new'] = '0'
    test_x['rname_new'][test_x['road_name'].str.contains('국도')] = 'a'
    test_x['rname_new'][test_x['road_name'].str.contains('지방도')] = 'a'
    test_x['rname_new'][test_x['road_name'].str.contains('로')] = 'b'
    test_x['rname_new'][test_x['road_name'].str.contains('교')] = 'c'
    test_x['rname_new'][test_x['road_name'].str.contains('국_지_도')] = 'a'
    #test_x['rname_new'][test_x['road_name'].str.contains('NaN')] = 'b'
    test_x['rname_new'][test_x['road_name'] == '0'] = 'a'

    ohe = OneHotEncoder(sparse=False)

    train_cat = ohe.fit_transform(train_x[['rname_new']])
    pd.DataFrame(train_cat, columns=['rname_new' + col for col in ohe.categories_[0]])
    train_x = pd.concat([train_x.drop(columns=['rname_new']), pd.DataFrame(train_cat, columns=['rname_new' + col for col in ohe.categories_[0]])], axis=1)


    # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
    test_cat = ohe.transform(test_x[['rname_new']])
    pd.DataFrame(test_cat, columns=['rname_new' + col for col in ohe.categories_[0]])
    test_x = pd.concat([test_x.drop(columns=['rname_new']), pd.DataFrame(test_cat, columns=['rname_new' + col for col in ohe.categories_[0]])], axis=1)

    return train_x, test_x

def weight_restricted_new(train_x,test_x):
    #제한이 클 수록 차들 평균 통행 속도가 빠른 곳일 것
    train_x['weight_restricted_new'] = 'a'
    #train_x['weight_restricted_new'][(train_x['weight_restricted'] == 0)] = 'a'
    train_x['weight_restricted_new'][(train_x['weight_restricted'] == 32400)] = 'b'
    train_x['weight_restricted_new'][(train_x['weight_restricted'] == 43200)] = 'c'
    train_x['weight_restricted_new'][(train_x['weight_restricted'] == 50000)] = 'b'

    train_x['weight_restricted_new'][(train_x['weight_restricted'] != 'a')&(train_x['weight_restricted'] != 'b')&(train_x['weight_restricted'] != 'c')] = 'a'

    test_x['weight_restricted_new'] = 'a'
    #test_x['weight_restricted_new'][(test_x['weight_restricted'] == 0)] = 'a'
    test_x['weight_restricted_new'][(test_x['weight_restricted'] == 32400)] = 'b'
    test_x['weight_restricted_new'][(test_x['weight_restricted'] == 43200)] = 'c'
    test_x['weight_restricted_new'][(test_x['weight_restricted'] == 50000)] = 'b'

    test_x['weight_restricted_new'][(test_x['weight_restricted'] != 'a')&(test_x['weight_restricted'] != 'b')&(test_x['weight_restricted'] != 'c')] = 'a'

    ohe = OneHotEncoder(sparse=False)

    train_cat = ohe.fit_transform(train_x[['weight_restricted_new']])
    pd.DataFrame(train_cat, columns=['weight_restricted_new' + col for col in ohe.categories_[0]])
    train_x = pd.concat([train_x.drop(columns=['weight_restricted_new']), pd.DataFrame(train_cat, columns=['weight_restricted_new' + col for col in ohe.categories_[0]])], axis=1)


    # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
    test_cat = ohe.transform(test_x[['weight_restricted_new']])
    pd.DataFrame(test_cat, columns=['weight_restricted_new' + col for col in ohe.categories_[0]])
    test_x = pd.concat([test_x.drop(columns=['weight_restricted_new']), pd.DataFrame(test_cat, columns=['weight_restricted_new' + col for col in ohe.categories_[0]])], axis=1)

    return train_x, test_x


############ day of week -> target값에 따라 그룹핑한거 ###############
def day_of_week_2(train_x,test_x):
    train_x['day_of_week_new'] = 'a'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '월')] = 'a'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '화')] = 'a'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '수')] = 'a'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '목')] = 'a'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '금')] = 'b'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '토')] = 'a'
    train_x['day_of_week_new'][(train_x['day_of_week'] == '일')] = 'c'

    train_x.drop(['day_of_week'], axis = 1, inplace = True)

    test_x['day_of_week_new'] = 'a'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '월')] = 'a'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '화')] = 'a'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '수')] = 'a'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '목')] = 'a'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '금')] = 'b'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '토')] = 'a'
    test_x['day_of_week_new'][(test_x['day_of_week'] == '일')] = 'c'
    ohe = OneHotEncoder(sparse=False)

    test_x.drop(['day_of_week'], axis = 1, inplace = True)
    
    train_cat = ohe.fit_transform(train_x[['day_of_week_new']])
    pd.DataFrame(train_cat, columns=['day_of_week_new' + col for col in ohe.categories_[0]])
    train_x = pd.concat([train_x.drop(columns=['day_of_week_new']), pd.DataFrame(train_cat, columns=['day_of_week_new' + col for col in ohe.categories_[0]])], axis=1)


    # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
    test_cat = ohe.transform(test_x[['day_of_week_new']])
    pd.DataFrame(test_cat, columns=['day_of_week_new' + col for col in ohe.categories_[0]])
    test_x = pd.concat([test_x.drop(columns=['day_of_week_new']), pd.DataFrame(test_cat, columns=['day_of_week_new' + col for col in ohe.categories_[0]])], axis=1)

    return train_x, test_x
    

def speed(train, test, col, col_name):
  speed = train.groupby([col, 'maximum_speed_limit'])['target'].agg([(col_name, 'mean')]).reset_index()
  train = pd.merge(train, speed, on=[col, 'maximum_speed_limit'], how='left')
  test = pd.merge(test, speed, on=[col, 'maximum_speed_limit'], how='left')
  return train, test

def jeju_dist(df):
    from haversine import haversine
    jeju_location = (33.4996213, 126.5311884)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine

def seogwi_dist(df):
    from haversine import haversine
    jeju_location = (33.2541205, 126.560076)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine

def make_log(df, col):
    import math
    return df[col].apply(lambda x: math.log(x))


def end_node_t(train,test):
    endnode_target=train.groupby(['end_node_name']).agg({'target':'mean'}).sort_values(by=['target']).reset_index()
    node_under_20 = endnode_target[endnode_target['target']<20]['end_node_name'].unique()
    node_under_40 = endnode_target[endnode_target['target']<40]['end_node_name'].unique()
    node_under_60 = endnode_target[endnode_target['target']<60]['end_node_name'].unique()
    node_under_max = endnode_target[endnode_target['target']>=60]['end_node_name'].unique()

    train['end_node_t_a']=train['end_node_name'].apply(lambda x: 1 if x in node_under_20  else 0)
    train['end_node_t_b']=train['end_node_name'].apply(lambda x: 1 if x in node_under_40  else 0)
    train['end_node_t_c']=train['end_node_name'].apply(lambda x: 1 if x in node_under_60  else 0)
    train['end_node_t_d']=train['end_node_name'].apply(lambda x: 1 if x in node_under_max  else 0)

    test['end_node_t_a']=test['end_node_name'].apply(lambda x: 1 if x in node_under_20  else 0)
    test['end_node_t_b']=test['end_node_name'].apply(lambda x: 1 if x in node_under_40  else 0)
    test['end_node_t_c']=test['end_node_name'].apply(lambda x: 1 if x in node_under_60  else 0)
    test['end_node_t_d']=test['end_node_name'].apply(lambda x: 1 if x in node_under_max  else 0)

def another(train_x,test_x):
    train_x['road_rating_ch'] = 0 
    train_x['road_rating_ch'][train_x['road_rating'] == 107] = 1
    train_x['road_rating_ch'][train_x['road_rating'] == 106] = 2
    train_x['road_rating_ch'][train_x['road_rating'] == 103] = 3

    test_x['road_rating_ch'] = 0
    test_x['road_rating_ch'][test_x['road_rating'] == 107] = 1
    test_x['road_rating_ch'][test_x['road_rating'] == 106] = 2
    test_x['road_rating_ch'][test_x['road_rating'] == 103] = 3
    
    train_x['lane_ms'] = train_x['lane_count']*train_x['maximum_speed_limit']
    test_x['lane_ms'] = test_x['lane_count']*test_x['maximum_speed_limit']

    train_x['lane_rating'] = train_x['lane_count'] * train_x['road_rating_ch']
    test_x['lane_rating'] = test_x['lane_count'] * test_x['road_rating_ch']

    train_x['rating_ms'] = train_x['road_rating_ch']*train_x['maximum_speed_limit']
    test_x['rating_ms'] = test_x['road_rating_ch']*test_x['maximum_speed_limit']


    #train_x['lazy_ms'] = train_x['maximum_speed_limit'] / (train_x['lazy_houra.1']+1)
    #test_x['lazy_ms'] = test_x['maximum_speed_limit'] / (test_x['lazy_houra.1']+1)

    #train_x['weight_ms'] = (train_x['weight_restricteda']+1)*train_x['maximum_speed_limit']
    #test_x['weight_ms'] = (test_x['weight_restricteda']+1)*test_x['maximum_speed_limit']

    train_x['move_latitude'] = abs(train_x['start_latitude']-train_x['end_latitude'])*1000
    train_x['move_longitude'] = abs(train_x['start_longitude']-train_x['end_longitude'])*1000

    test_x['move_latitude'] = abs(test_x['start_latitude']-test_x['end_latitude'])*1000
    test_x['move_longitude'] = abs(test_x['start_longitude']-test_x['end_longitude'])*1000

    train_x['node_tf']=train_x['start_node_name']==train_x['end_node_name']
    test_x['node_tf']=test_x['start_node_name']==test_x['end_node_name']
    #위경도 scaling
    train_x['start_latitude'] = train_x['start_latitude'] - 33
    train_x['end_latitude'] = train_x['end_latitude'] - 33
    train_x['start_longitude'] = train_x['start_longitude'] - 126
    train_x['end_longitude'] = train_x['end_longitude'] - 126

    test_x['start_latitude'] = test_x['start_latitude'] - 33
    test_x['end_latitude'] = test_x['end_latitude'] - 33
    test_x['start_longitude'] = test_x['start_longitude'] - 126
    test_x['end_longitude'] = test_x['end_longitude'] - 126

    train_x['connect_code'][(train_x['connect_code'] == 103)] = 1
    test_x['connect_code'][(test_x['connect_code'] == 103)] = 1

def MinMax(train_x,test_x):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    train_x['start_latitude_MinMax'] = scaler.fit_transform(train_x['start_latitude'].values.reshape(-1,1))
    test_x['start_latitude_MinMax'] = scaler.transform(test_x['start_latitude'].values.reshape(-1,1))

    train_x['start_longitude_MinMax'] = scaler.fit_transform(train_x['start_longitude'].values.reshape(-1,1))
    test_x['start_longitude_MinMax'] = scaler.transform(test_x['start_longitude'].values.reshape(-1,1))

    train_x['end_latitude_MinMax'] = scaler.fit_transform(train_x['end_latitude'].values.reshape(-1,1))
    test_x['end_latitude_MinMax'] = scaler.transform(test_x['end_latitude'].values.reshape(-1,1))

    train_x['end_longitude_MinMax'] = scaler.fit_transform(train_x['end_longitude'].values.reshape(-1,1))
    test_x['end_longitude_MinMax'] = scaler.transform(test_x['end_longitude'].values.reshape(-1,1))

def sm_tm(train, test):    
    a = train.groupby('maximum_speed_limit')['target'].agg([('sm_tm', 'mean')]).reset_index()
    a['diff'] = a['maximum_speed_limit'] - a['sm_tm']
    a = a.drop(['sm_tm'], axis=1)
    train = pd.merge(train, a, on=['maximum_speed_limit'], how='left')
    test = pd.merge(test, a, on=['maximum_speed_limit'], how='left')
    return train, test


def node_tf(train_x,test_x):
    train_x['node_tf']=train_x['start_node_name']==train_x['end_node_name']
    test_x['node_tf']=test_x['start_node_name']==test_x['end_node_name']
    str_col = ['node_tf','end_node_name','start_node_name']
    for i in str_col:
        le = LabelEncoder()
        le=le.fit(train_x[i])
        train_x[i]=le.transform(train_x[i])
        
        for label in np.unique(test_x[i]):
            if label not in le.classes_: 
                le.classes_ = np.append(le.classes_, label)
        test_x[i]=le.transform(test_x[i])



import pandas as pd
#from feature.feature_selection import *
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def make_dataset(train, test):    
    start = datetime.now()
    print('Start time: ', start)
    
    


    # 2022 07 기준으로 target 수정
    dif=train[train['base_date']<20220700]['target'].mean()-train[train['base_date']>20220700]['target'].mean()
    tmp1=train[train['base_date']<20220700]
    tmp1['target']=tmp1['target']-dif
    tmp2=train[train['base_date']>20220700]
    train=pd.concat([tmp1,tmp2]).reset_index()

    train=train[train['maximum_speed_limit'] !=40].reset_index()

    train, test = group_season(train, test)
    cyclical_feature(train)
    train['month'] = make_month(train)
    train['week'] = make_week(train)
    train['week'] = train.apply(week_mapping, axis=1)
    train['distance'] = make_dist(train)
    train['turn_restricted'] = turn_restricted(train)
    #move_lat_lng(train)
    #interval(train)
    #lat_lng_scale(train)
    train['jeju_dist'] = jeju_dist(train)
    train['seogwi_dist'] = seogwi_dist(train)
    #train['base_hour_2'] = base_hour_2(train)
    print('Train dataset success !')


    cyclical_feature(test)
    test['month'] = make_month(test)
    test['week'] = make_week(test)
    test['week'] = test.apply(week_mapping, axis=1)
    test['distance'] = make_dist(test)
    test['turn_restricted'] = turn_restricted(test)
    #move_lat_lng(test)
    #interval(test)
    #lat_lng_scale(test)
    test['jeju_dist'] = jeju_dist(test)
    test['seogwi_dist'] = seogwi_dist(test)
    #test['base_hour_2'] = base_hour_2(test)
    print('Test dataset success !')

    train, test = make_cluster(train, test)
    train, test = speed(train,test,'end_node_name','end_speed')
    train, test = speed(train,test,'start_node_name','start_speed')
    train, test = speed(train,test,'road_name','section_speed')
    train, test = sm_tm(train,test)
    train, test = group_time(train, test)
    lat_lng_minmax(train, test)
    train, test = road_name_new(train,test)
    train, test = weight_restricted_new(train, test)
    train, test = day_of_week_2(train, test)
    train, test = over_max_speed(train,test)
    another(train,test)
    MinMax(train,test)
    #move_lat_lng(train)
    #move_lat_lng(test)
    node_tf(train,test)
    interval(train)
    interval(test)

    lat_lng_scale(train)
    lat_lng_scale(test)

    lat_lng_minmax(train, test)

    train['dis_log'] = make_log(train,'distance')
    test['dis_log'] = make_log(test,'distance')
    # candidate = ['target']
    # for cand in candidate:  
    #     train = remove_outlier(train,cand)
    
    # train.reset_index(drop = True,inplace = True)
    
    X = train.drop(
    ['id', 'road_in_use','target', 'height_restricted','base_date', 'road_name', 'start_node_name', 'end_node_name', 'vehicle_restricted', 'base_hour'], axis=1
    )

    y = train['target']

    test = test.drop(
        ['id', 'road_in_use', 'height_restricted','base_date', 'road_name', 'start_node_name', 'end_node_name', 'vehicle_restricted', 'base_hour'], axis=1
    )

    End = datetime.now()
    print(f'End time: {End}' )
    print('Play time: ', End - start)
    
    return X, y, test

# train = pd.read_csv('./train.csv')
# test_x = pd.read_csv('./test.csv')

# train_x, train_y, test_x = make_dataset(train, test_x)


# train_x.to_csv("After_Team_train_x2.csv", index = False)
# train_y.to_csv("After_Team_train_y2.csv", index = False)
# test_x.to_csv("After_Team_test_x2.csv", index = False)
