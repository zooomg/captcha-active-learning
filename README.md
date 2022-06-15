# captcha-active-learning

2022-1 빅데이터처리 과목 Active learning 텀프로젝트
- 문제 설정: [Large Captcha Dataset](https://www.kaggle.com/datasets/akashguna/large-captcha-dataset)에 active learning 기법 적용해보기
- 데이터셋
  - 5개의 문자를 나타내는 image
  - 숫자, 알파벳 대문자 및 소문자 62개 중 하나의 문자, 중복 허용
  - 82,328장의 image로 구성
  - image의 파일명이 image에 적혀있는 text
  - image에 Salt-and-pepper noise와 line noise 존재
<img width="450" alt="image" src="https://user-images.githubusercontent.com/11240557/173719699-5759207f-8394-4212-b98a-bf77bda3c838.png">

## Preprocessing
- [Deep-CAPTCHA: a deep learning based CAPTCHA solver for vulnerability assessment](https://arxiv.org/abs/2006.08296)을 참고
- 이미지의 상하 여백을 지우기 위한 crop 적용
- noise를 지우기 위한 median-filter 적용


<img width="1259" alt="image" src="https://user-images.githubusercontent.com/11240557/173720359-fbde9a7d-c168-473e-8d94-17dfb08d8597.png">

## Model
- 62개의 문자를 순서를 고려하여 한꺼번에 맞출 경우 62^5 중 하나를 선택하는 문제
- 보다 단순하게 만들기 위해 각 자리를 예측하는 fully connected layer 5개 사용
- 62^5에서 5x62로 경우의 수 감소
- image feature를 학습하기 위해 ResNet18 사용

<img width="1255" alt="스크린샷 2022-06-15 오전 11 07 04" src="https://user-images.githubusercontent.com/11240557/173721177-b47b0c64-802a-4ff4-9f6f-8e38f4b3c383.png">

## Method
- 사용한 active learing 기법은 uncertainty pool-based method
  - 라벨이 되지않은 데이터 풀(unlabeled pool)에서 uncertainty가 높은 데이터를 뽑아서 사람(oracle)에게 라벨링 요청 후
  - 라벨이 된 데이터 풀(labeled pool)에 넣어서 fine-tuning 진행

<img width="80%" alt="image" src="https://user-images.githubusercontent.com/11240557/173721511-29a056ce-3857-4ca2-bf3f-d1d50915ab5a.png">

- uncertainty를 측정하는 방법은 다음과 같이 사용
  - 흔히 사용되는 3가지 기법
    - Least Confidence
      - 예측한 class 중 가장 높은 logit으로 예측한 값이 가장 작은 sample을 뽑기
    - Margin Sampling
      - 예측한 class 중 가장 높은 logit과 두 번째로 큰 logit 값의 차이가 작은 sample 뽑기
    - Entropy
      - entropy 값이 높은 sample 뽑기
  - [Deep-CAPTCHA: a deep learning based CAPTCHA solver for vulnerability assessment](https://arxiv.org/abs/2006.08296)에서 소개된 기법
    - Best versus Second Best
      - 예측한 class 중 (가장 높은 logit / 두 번째로 큰 logit) 값이 가장 큰 sample 뽑기
  - 우리가 만든 기법
    - Pick Most Even Sample
      - margin sampling과 entropy를 섞어보려고 했음
      - 가장 높은 logit과 두 번째로 큰 logit 값의 차이가 작은 + 가장 높은 logit과 가장 작은 logit 값의 차이가 작은 sample 뽑기
  - Base
    - Random Sampling

- image 하나 당 5개의 confidence가 구해지는데 이를 하나로 만들 방법
  - mean
  - median

## Result

### 5-epoch, 29-iter, 2000-sample
- 학습이 너무 많이 진행되다보니 overfitting 발생
- 학습 후반부에 acc가 요동치는 현상 확인
- 또한, active learning을 적용하지 않고 모든 데이터를 활용한 학습하여 얻은 base acc에 도달하지 못함
- 각 기법으로 뽑은 sample의 영향이 크게 작용하여 random보다 다른 기법의 성능이 높다는 사실 확인

<img width="215" alt="image" src="https://user-images.githubusercontent.com/11240557/173723006-ed2be6a2-fb7a-48a6-b5ed-3e4e4db485f4.png"><img width="224" alt="image" src="https://user-images.githubusercontent.com/11240557/173723017-354a8d36-2159-4295-9cb3-d7507878c32f.png"><img width="202" alt="image" src="https://user-images.githubusercontent.com/11240557/173723090-1f3b899a-7d3c-4365-97ad-64876eae7374.png"><img width="206" alt="image" src="https://user-images.githubusercontent.com/11240557/173723096-6c8d12ec-2e05-432f-8fbf-f7714a028b69.png"><img width="192" alt="image" src="https://user-images.githubusercontent.com/11240557/173723105-c5500da0-dae7-43e6-bfc5-94cab3166f64.png">

### 2-epoch, 15-iter, 4000-sample
- overfitting을 방지하고자 총 학습되는 수를 줄여서 실험 진행
  - base acc를 얻는데 필요한 epoch 수가 25여서 이와 유사하게 epoch과 iter를 세팅
- acc는 안정되고 base acc에 도달
- 그러나 각 기법으로 뽑은 sample의 영향력이 줄어들어 random에 비해 다른 기법이 월등히 나은 결과를 보이지 못한다고 추측

<img width="225" alt="image" src="https://user-images.githubusercontent.com/11240557/173723220-e653e361-991b-4230-a634-830a72e8eee0.png"><img width="227" alt="image" src="https://user-images.githubusercontent.com/11240557/173723226-162fb2bf-f7bd-40be-8186-96cc36ebc0db.png"><img width="224" alt="image" src="https://user-images.githubusercontent.com/11240557/173723239-6bc7ab57-42b8-4592-aade-43b9ffce9817.png"><img width="226" alt="image" src="https://user-images.githubusercontent.com/11240557/173723249-b781d0b6-784d-49a2-a046-36d1794f9db7.png"><img width="224" alt="image" src="https://user-images.githubusercontent.com/11240557/173723256-73e6fb79-f35a-416b-872a-78b89b23ba1b.png">

## Conclusion
- hyper-parameter(epoch, iter, sample)에 황금 비율이 있을 거라 예상하지만 텀프 기간 동안 찾지 못함
- real world에서 active learing의 적용이 쉽지 않을거라 판단
  - 전체 데이터셋에 대한 base acc가 어떻게 되는지 모름
  - 따라서 최적의 epoch 수를 찾기 힘듦
- sample 된 데이터의 영향이 클수록 random sampling에 비해 효과적일거라 예상
  - 그렇다고 epoch과 iter를 크게 줄 경우 overfitting 발생






