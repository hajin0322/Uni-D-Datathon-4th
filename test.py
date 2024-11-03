import os
from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from model import Restormer
from PIL import Image

# 랜덤 시드 고정 (재현성을 위해)
np.random.seed(42)

# 이미지 로드 함수 (OpenCV 사용)
def load_img(filepath):
    img = cv2.imread(filepath)  # 이미지를 BGR 형식으로 로드
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB로 변환
    return img

# Restormer 모델 로드
model = Restormer()
model.load_state_dict(torch.load('best_Restormer_400_final.pth'))  # 사전 학습된 모델 가중치 불러오기

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 데이터 경로 설정
noisy_data_path = './data/Test/noisy'  # 노이즈 이미지 경로
output_path = './data/Test/output'  # 결과 이미지 저장 경로

# 출력 경로가 존재하지 않으면 디렉토리 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Custom Dataset 클래스 정의 (테스트용)
class CustomDatasetTest(data.Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        # 이미지 경로 리스트 생성
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 인덱스에 해당하는 이미지 파일 경로 가져오기
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])  # 이미지 로드
        
        # numpy array -> PIL 이미지 변환
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        # 이미지 변환 적용 (있을 경우)
        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path  # 이미지와 경로 반환

# 이미지 전처리 파이프라인 정의
test_transform = Compose([
    transforms.CenterCrop(224),  # 중앙 부분만 자르기
    transforms.Resize((224, 224)),  # 크기 조정
    ToTensor(),  # 텐서로 변환
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

# 데이터셋 로드
noisy_dataset = CustomDatasetTest(noisy_data_path, transform=test_transform)

# 데이터 로더 설정
noisy_loader = DataLoader(noisy_dataset, batch_size=1, shuffle=False)

# 이미지 denoising 및 저장 루프
for noisy_image, noisy_image_path in noisy_loader:
    noisy_image = noisy_image.to(device)  # 이미지 데이터를 GPU로 이동
    denoised_image = model(noisy_image)  # 모델에 입력하여 denoising 수행
    
    # denoised_image를 CPU로 이동하여 이미지로 변환
    denoised_image = denoised_image.cpu().squeeze(0)  # 첫 번째 차원을 제거하여 이미지 형태로 만듦
    denoised_image = torch.clamp(denoised_image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
    denoised_image = transforms.ToPILImage()(denoised_image)  # PIL 이미지로 변환

    # Denoised 이미지 저장
    output_filename = noisy_image_path[0]
    denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.png'  # 저장 경로 설정
    denoised_image.save(denoised_filename)  # 이미지 저장
    
    print(f'Saved denoised image: {denoised_filename}')  # 저장 완료 메시지 출력

def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Created {output_zip}.zip successfully.")

zip_folder(output_path, './submission')
