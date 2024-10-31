# UNI-D 4th Baseline - Image Denoising Task Using Restormer

## 프로젝트 구조

프로젝트는 이미지 노이즈 제거(Image Denoising)를 위한 Restormer 모델을 구현하고, 학습 및 추론을 수행하는 파일들로 구성되어 있습니다.

```
${PROJECT}
├── README.md
├── preprocess.py
├── model.py
├── train.py
└── test.py
```

## 파일 설명

### 1. `preprocess.py`
- 데이터 전처리 파일입니다.
- 압축된 데이터셋 파일을 **unzip**하고, 모델 학습에 적합한 형태로 데이터 경로를 수정하는 작업을 수행합니다.

### 2. `model.py`
- Restormer 모델을 구현한 파일입니다.
- 이미지 노이즈 제거를 위한 모델의 구조와 세부 기능이 정의되어 있습니다.

### 3. `train.py`
- Restormer 모델의 **학습**을 위한 파일입니다.
- 전처리된 데이터를 사용하여 모델을 학습시키는 코드를 포함하고 있습니다.

### 4. `test.py`
- 이미지 노이즈 제거를 위한 **추론(Inference)** 파일입니다.
- 학습된 모델을 이용하여 노이즈가 포함된 이미지에 대해 노이즈 제거 작업을 수행하는 코드를 포함하고 있습니다.

---

각 파일의 역할에 맞게 프로젝트를 구성하고 필요한 작업을 수행해 주세요.