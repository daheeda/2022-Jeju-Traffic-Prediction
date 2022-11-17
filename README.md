# 제주도 도로 교통량 예측 AI 경진대회

<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">제주도 교통량 예측 </h3>

<div align="center">
  
  ![Python Version](https://img.shields.io/badge/Python-3.8.10-blue)
</div>

---
## 🧐 About <a name = "about"></a>
제주도 도로 교통량 예측 AI 알고리즘 개발  
제주도의 교통 정보로부터 도로 교통량 회귀 예측

## 🖥️ Development Environment
```
OS: Window11
CPU: Intel i9-11900K
RAM: 128GB
GPU: NVIDIA GeFocrce RTX3090
```

## 🔖 Project structure

```
Project_folder/
|- EDA/          # eda (ipynb)
|- feature/      # feature engineering (py)
|- garbage/      # garbage 
|- jeju_data     # required data (csv & parquet)
|- model/        # model test by feature (ipynb)
|- reference/    # paper (pdf)
|- main.py       # final model (py)
```

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Setup your environement and install project dependencies
```
python -m venv project
project\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```



## 📖 Dataset
**Data Source**  
[Train Test Dateset](https://dacon.io/competitions/official/235985/overview/description) &nbsp;&nbsp; [Tour](https://www.data.go.kr/data/15004770/fileData.do)  
```
Dataset Info.

train.csv (4701217, 49)
2022년 8월 이전 데이터만 존재 (단, 날짜가 모두 연속적이지 않음)
id : 샘플 별 고유 id
날짜, 시간, 교통 및 도로구간 등 정보
target : 도로의 차량 평균 속도(km)

test.csv (291241, 48)
2022년 8월 데이터만 존재 (단, 날짜가 모두 연속적이지 않음)
id : 샘플 별 고유 id
날짜, 시간, 교통 및 도로구간 등 정보

국가공휴일.csv
2018 ~ 2023년의 국가 공휴일

Tour.csv
제주도 장소데이터
공항, 항만, 아파트, 마트, 관광지, 학교 등
```



## 🔧 Feature Engineering
**관광지**라는 특성을 중점으로 feature 생성
```
Feature selection

시간 (0 ~ 24)
1. cyclical_feature: 24시간을 주기성을 가지는 데이터로 변환
2. group_time: 새벽, 아침, 점심, 저녁

요일(week)
1. make_week: 공휴일
2. make_holiday: 주말
3. make_post_holiday, make_pre_holiday: 전날이 공휴일, 다음날이 공휴일
4. rest: 주말, 공휴일 

날짜(Ymd)
1. make_month: 달
2. group_season: 봄, 여름, 가을, 겨율
3. vacation: 방학(7~8 & 12~2)

위도, 경도
1. make_dist: 두 지점 사이의 거리
2. make_cluster: 지역변수로 clustering => 4가지로 분류
3. * dist: 관광지와 끝지점으로 부터의 거리
4. Tour_count: 2km내 관광지 개수

Target
1. maximum_speed_limit & road: 속도제한을 고려한 Target 평균값
2. time & road: 시간대를 고려한 Target 평균값

Other
1. turn_restricted: 시작, 끝의 회전제한 유무
2. node_tf: 시작, 끝지점이 같은지 
3. sm_tm: maximum_speed_limit - mean(target)
4. road_name_set: road_name 집합


Encoding
1. Labelencoder
str_col = ['day_of_week', 'start_turn_restricted', 'end_turn_restricted', 'road_name', 'start_node_name',  'end_node_name','group_time', 'season', 'vacation' 'road_name_set', 'end_cartesian']


Drop Feature
drop = ['id', 'base_date', 'target', 'vehicle_restricted', 'height_restricted', 'post_date', 'pre_date']
```



## 🎈 Modeling

**Model**
```
XGBoost
Catboost
LGBM
LSTM
NN
AutoML: Autogluon, pycarat
```
**HyperParameter Tuning**
```
Optuna
```
**Cross Validation**
```
StratifiedKFold
```
**Ensemble**
```
Stacking
Blending
Voting
```

##  ✍️ Authors
### **Leader**
- ``전주혁`` [@ jjuhyeok](https://github.com/jjuhyeok)

### **Member**
- ``곽명빈`` [@ Myungbin](https://github.com/Myungbin?tab=repositories)
- ``박재열`` [@ hitpjy](https://github.com/hitpjy)
- ``최다희`` [@ Dahee Choi](https://github.com/daheeda)
- ``최새한`` [@ saehan](https://github.com/saehan-choi)



