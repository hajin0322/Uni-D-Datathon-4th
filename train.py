import os
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToPILImage, ToTensor
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Restormer
from torchvision import transforms  # transforms 모듈 추가
from torchvision.transforms import CenterCrop, Resize
from PIL import Image
from tqdm import tqdm
import math

'''
1. 모델 하이퍼파라미터 수정(학습률, 배치 크기, 에폭 수 등)
2. _데이터 증강 추가 또는 수정_
3. 학습 스케줄러 수정 (CosineAnnealing, StepLR 등)
4. 손실 함수 및 최적화 방법 수정
5. 다양한 모델 실험(모델 구조 변경 시 model.py의 일부를 참고)
'''

# PSNR 계산 함수
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

# 시작 시간 기록 (훈련 시간 측정)
start_time = time.time()

# 가중치 초기화 함수(모델 학습 초기 상태에서 좋은 분포로 가중치 설정)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

# 이미지 로드 함수(OpenCV통해 RGB 형식으로 변환)
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


from torchvision.transforms import GaussianBlur
import torchvision.transforms.functional as TF

# 가우시안 블러와 노이즈 추가를 위한 사용자 정의 함수
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# 커스텀 데이터셋 클래스 정의(데이터셋 정의, noisy이미지와 clean이미지를 받아 쌍으로 처리)
class CustomDataset(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        # 각 이미지의 경로 목록 생성
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(224)
        self.resize = Resize((224, 224))

        # (노이즈, 깨끗한 이미지) 쌍 생성
        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    # 노이즈 이미지와 대응하는 깨끗한 이미지 쌍 생성
    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path
        
        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
        
        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]

        # 이미지를 RGB로 로드
        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")
        
        # 이미지 중앙 크롭 및 리사이즈
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)
        
        # 전처리 변환 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

# 모델의 파라미터 수를 계산하는 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 하이퍼파라미터 설정
num_epochs = 11 # 학습할 에폭 수
batch_size = 8 # 한 번에 처리할 이미지 묶음 크기
learning_rate = 0.0005 # 학습률

# 데이터셋 경로 설정
noisy_image_paths = './data/Training/noisy'
clean_image_paths = './data/Training/clean'
val_noisy_image_paths = './data/Validation/noisy'
val_clean_image_paths = './data/Validation/clean'


# 데이터 전처리 파이프라인 정의
train_transform = Compose([
    
    transforms.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),  # 가우시안 블러를 50% 확률로 적용
    ToTensor(),
    # 텐서 변환 (3차원 배열)
])
# 데이터셋 로드
train_dataset = CustomDataset(clean_image_paths, noisy_image_paths, transform=train_transform)

# 데이터 로더 설정 (데이터셋 로드 및 병렬 처리)
num_cores = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

# GPU 사용 여부 확인 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Restormer 모델 인스턴스 생성 및 GPU로 이동
model = Restormer().to(device)




# 손실 함수와 최적화 알고리즘 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # 가중치 업데이트
criterion = nn.L1Loss()  # 예측된 이미지와 깨끗한 이미지의 차이를 절대값으로 측정
scaler = GradScaler()  # Mixed Precision 훈련을 위한 GradScaler 설정(메모리 사용량 감소)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)  # 학습률 스케줄러 설정(학습이 진행될수록 학습률 감소)



# 모델의 파라미터 수 출력
total_parameters = count_parameters(model)
print("Total Parameters:", total_parameters)

# 모델 학습 루프
model.train()
best_loss = 1000

# GPU 사용량 확인
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"GPU 사용 가능. GPU 수: {device_count}")
    for i in range(device_count):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i} - 이름: {gpu_properties.name}, 메모리: {gpu_properties.total_memory}MB")
        allocated_memory = torch.cuda.memory_allocated() / 1024**2  # MB 단위로 변환
        print(f"현재 할당된 GPU 메모리: {allocated_memory:.2f} MB")
else:
    print("GPU 사용 불가능")

# Epoch 별 학습
for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    mse_running_loss = 0.0
    
    # 미니배치 반복 (TQDM을 이용해 진행률 표시)
    for noisy_images, clean_images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        
        optimizer.zero_grad()  # 기울기 초기화
        
        # Mixed Precision Training
        with autocast():
            outputs = model(noisy_images)  # 모델 예측
            mse_loss = criterion(outputs, clean_images)  # 손실 계산
        
        # 손실을 역전파하고 가중치 업데이트
        scaler.scale(mse_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 기울기 클리핑
        scaler.step(optimizer)
        scaler.update()
        
        mse_running_loss += mse_loss.item() * noisy_images.size(0)

    # 학습률 업데이트 (Epoch이 끝난 후에 한번씩 호출)
    scheduler.step()

    # GPU 캐시 비우기
    torch.cuda.empty_cache()
        
    mse_epoch_loss = mse_running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, MSE Loss: {mse_epoch_loss:.4f}')
    
    # 모델 저장 부분 수정: 손실이 줄어들 때마다 저장
    if mse_epoch_loss < best_loss:
        best_loss = mse_epoch_loss
        model_filename = f'best_Restormer_epochh{epoch+1}_loss{mse_epoch_loss:.4f}.pth'
        torch.save(model.state_dict(), model_filename)
        print(f"{epoch+1}epoch 모델 저장 완료: {model_filename}")




# 평가 데이터셋 로드
val_dataset = CustomDataset(val_clean_image_paths, val_noisy_image_paths, transform=train_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 모델 평가 (PSNR 계산)
model.eval()
total_psnr = 0
count = 0

with torch.no_grad():
    for noisy_images, clean_images in tqdm(val_loader, desc="Evaluating"):
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        
        outputs = model(noisy_images)
        
        # PSNR 계산
        psnr = calculate_psnr(outputs, clean_images)
        total_psnr += psnr
        count += 1

average_psnr = total_psnr / count
print(f"Average PSNR on validation set: {average_psnr:.2f} dB")


# 전체 훈련 소요 시간 출력
end_time = time.time()
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")
