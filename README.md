# 🚘 Hyundai_Motor

- joint research about DTC prediction(Anomaly Detection)
- exact code is in Hyundai VDI (can't afford it)

### common preprocessing
1️⃣ 모든 데이터 수집 버전이 보유한 공통 변수 추출    
2️⃣ 고장이 발생했지만 처음부터 끝까지 변동이 존재하지 않는 변수 삭제  
3️⃣ 날씨 및 계절에 영향받는 변수 고려해 운행된 차량별로 스케일링 진행  
4️⃣ 다중공선성 방지하기 위해 변수간 상관계수(pearson, cramers'V)를 구한 후 상관계수 기준(0.9,0.8) DFS 알고리즘으로 그룹핑 진행  
5️⃣ 그룹 내 edge 가장 많은 변수만 변수 선택

### hyundai_LGBM
- captured image of original code which located in VDI
- Solution with LGBM  
  1️⃣ 트리 모델 3가지를 통해 변수 중요도 산정 후 상위 40개씩 합집합(불필요한 변수가 들어가면서 적당한 노이즈 효과를 주어 일반화 성능 향상)  
  2️⃣ 정상 0, 전조 1, 고장 2 라벨링 전조 시간을 4시간부터 12시간까지 1시간 단위로 변경해가며 라벨링 후 검증(자동화 되어있음)  
  3️⃣ 전조 시간을 4시간부터 12시간까지 1시간 단위로 변경해가며 라벨링 후 검증(자동화 되어있음)optuna를 통해 parameter 최적화  
  4️⃣ optuna를 통해 parameter 최적화  
  5️⃣ 예측 결과를 휴리스틱 방식으로 resample하여 성능 최적화

### hyundai_AD
- Solution with Anomaly Transformer  
  1️⃣ 파생 변수 생성(resample로 sampling rate 통일할 때 mean, std, mse, min, max 등 생성)  
  2️⃣ 변수 개수 줄이기 위해 파상변수끼리 PCA 0.9 진행  
  3️⃣ 연구 목표 및 데이터 특성 고려해 Prior Association 강화(softmax와 비슷한 작업으로 합 1 수치로 변경할 때 temperture 낮춤)  
  4️⃣ 산출된 Anomaly Score를 휴리스틱 방식으로 임계값 설정(일반화 실패)

### 연구 결과
- 딥러닝 layer 쌀일수록 정상과 고장의 reconstruction error 차이 증가
- LightGBM으로 평균 F1 score 0.831 달성
- 논문 '[차량 DTC 고장 예측을 위한 딥러닝 적용 사례 연구, CDE,2023](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11513761&nodeId=NODE11513761&medaTypeCode=185005&locale=ko&foreignIpYn=N&articleTitle=%EC%B0%A8%EB%9F%89+DTC+%EA%B3%A0%EC%9E%A5+%EC%98%88%EC%B8%A1%EC%9D%84+%EC%9C%84%ED%95%9C+%EB%94%A5%EB%9F%AC%EB%8B%9D+%EC%A0%81%EC%9A%A9+%EC%82%AC%EB%A1%80+%EC%97%B0%EA%B5%AC&articleTitleEn=A+Case+Study+on+applying+Deep+Learning+Methods+to+Predict+Vehicle+DTC+Faults&language=ko_KR&hasTopBanner=true)' 작성

