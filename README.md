# 🚘 Hyundai_Motor

## joint research about DTC prediction(Anomaly Detection)
- exact code is in Hyundai VDI (can't afford it)

## 대회 개요
+ **주최/주관** | 산업통상자원부, 대구광역시/ 한국자동차연구원, 대구디지털혁신진흥원
+ **운영** | [데이콘](https://dacon.io/competitions/official/236193/overview/description)
+ **기간** | 2023.11.15 ~ 2023.12.11
+ **주제** | 시공간 정보로부터 사고위험도(ECLO) 예측 AI 모델 개발
  
## 목표
사고 발생 시간, 공간 등의 정보를 활용하여 사고위험도(ECLO)를 예측하는 AI 알고리즘 개발

## 결과
- **최종 1위 / 941팀 (대상) 🏆**
- Public Score 0.4254763748 | Private Score 0.42547

## DataSet

**INPUT - [대회 제공 내부 데이터](https://dacon.io/competitions/official/236193/data)**
+ **ID** : 대구에서 발생한 교통사고의 고유 ID
+ **시간 정보** : 사고일시, 요일
+ **공간 정보** : 시군구
+ **환경 정보** : 기상상태, 도로형태, 노면상태, 사고유형, 사고유형-세부분류, 법규 위반
+ **가해 운전자 정보** : 가해운전자 차종, 가해운전자 성별, 가해운전자 연령, 가해운전자 상해정도
+ **피해 운전자 정보** : 피해운전자 차종, 피해운전자 성별, 피해운전자 연령, 피해운전자 상해정도

**INPUT - 외부 데이터**
+ [천문역법참조표준데이터_20161202](https://www.data.go.kr/data/15053554/fileData.do)
+ [전국 노드링크별 평균택시통행량_20211231](https://www.data.go.kr/data/15069016/fileData.do)
+ [전국 노드링크별 평균택시속도_20211231](https://www.data.go.kr/data/15069020/fileData.do)

**TARGET**
+ **ECLO(Equivalent Casualty Loss Only)** : 인명피해 심각도
  + ECLO = 사망자수 * 10 + 중상자수 * 5 + 경상자수 * 3 + 부상자수 * 1

## Model
- `XGBoostRegressor`, `CatBoostRegressor`, `LightGBMRegressor`

## Feature Engineering
`test.csv`에 제공되지 않은 사고유형-세부분류, 법규위반, 가해운전자 상해정도, 피해운전자 상해정도 Feature 삭제  

**내부 데이터 파생 변수**
| 파생변수  | 설명 |
|:---:|:---|
|주말|요일 변수가 토요일, 일요일인 경우 인명피해 정도가 높아 주말 변수 추가|
|가해운전자 평균연령|운전자 평균연령과 ECLO의 양의 상관관계를 보이므로 지역별 가해운전자 평균 연령 추가|
|피해운전자 평균연령|운전자 평균연령과 ECLO의 양의 상관관계를 보이므로 지역별 피해운전자 평균 연령 추가|
|가해운전자 평균성별|운전자 성별에 따른 ECLO 차이를 반영하기 위하여 지역별 가해운전자 평균 연령 추가|
|피해운전자 평균성별|운전자 성별에 따른 ECLO 차이를 반영하기 위하여 지역별 피해운전자 평균 연령 추가|
|ride_dangerous|지역별로 가해 운전자 차종별 ECLO 차이를 반영하기 위하여 차종별 위험도 파생변수 추가|
|accident_case_dangerous|지역별로 사고유형별 ECLO 차이를 반영하기 위하여 사고유형별 위험도 파생변수 추가|

**외부 데이터 파생 변수**
| 파생변수  | 설명 |
|:---:|:---|
|sun|일출 일몰 시각에 따른 ECLO 차이를 반영하기 위하여 일출 일몰 시각 파생변수가|
|총통행량|지역별 교통환경 차이를 반영하기 위하여 구별 택시 총통행량 변수 생성|
|평균통행량|지역별 교통환경 차이를 반영하기 위하여 구별 택시 평균통행량 변수 생성|
|평균속도|지역별 교통환경 차이를 반영하기 위하여 구별 택시 평균속도 변수 생성|

## Insight
1️⃣ 지역별 택시의 평균 속도, 통행량은 ECLO에 영향을 미칩니다.  
2️⃣ 지역마다 사고유형별 사고 빈도와 가해운전자 차종별 사고 빈도는 ECLO와 연관성을 보입니다.  
3️⃣ 지역별 피해운전자 평균연령과 ECLO는 양의 상관관계를 보입니다.  
4️⃣ 야간의 시인성 저하는 교통사고 발생에 영향을 미칩니다.  
5️⃣ 인명 피해 정도는 평일보다 주말에 심해집니다.  
+ 자세한 내용은 [PPT자료](https://github.com/kkumtori/DACON-Traffic-Accident-Damage-Prediction-AI-Competition/blob/main/(%EC%B5%9C%EC%A2%85%EB%B3%B8)%ED%8C%80%20BITAmin%2013%EA%B8%B0%20%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf)에서 확인할 수 있습니다.

## Team Members
[노지예](https://github.com/kkumtori), [정세웅](https://github.com/eric981218), [조성우](https://github.com/jswooo), [임홍주](https://github.com/hihongju)

## Reference
- 야간의 시인성 저하가 교통사고에 미치는 영향 진단-경기도 지역의 경부, 서해안, 영동, 서울외곽순환고속도로를 중심으로-
- B. S. Chae, C. P. Han, “Research on the characteristics of a weekend traffic accident”, Korean Society of Civil Engineers Conference, pp.17-20, 10. 2009.
- J. S. Kim, K. H. Kim, B. H. Park, “Analyzing the Characteristics of Traffic Accidents and Developing the Accident Models on the Arterial Link Sections in Case of Cheongju”, 2010 Korean Society of Road Engineerings conference, pp.166-177, 3. 2010.

