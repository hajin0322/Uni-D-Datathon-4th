import os
import shutil

# Training 데이터셋을 전처리하는 함수
def preprocess_training(base_dir):
    # 'clean'과 'noisy' 디렉토리 경로 설정
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    # 'clean'과 'noisy' 디렉토리가 없으면 생성
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # 'GT'가 포함된 디렉토리를 찾기 위한 리스트 초기화
    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            # 디렉토리 이름에 'GT'가 포함된 경우 해당 디렉토리를 source_dirs에 추가
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    # 'GT'가 포함된 디렉토리가 없는 경우 오류 발생
    if not source_dirs:
        raise ValueError("No directory containing 'GT' found")

    # 'GT'가 포함된 디렉토리의 모든 .jpg 파일을 'clean' 폴더로 이동
    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    # 'clean' 및 'noisy' 디렉토리를 제외한 나머지 디렉토리에서 .jpg 파일을 찾아 'noisy' 폴더로 이동
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    # 'clean'과 'noisy' 디렉토리를 제외한 나머지 디렉토리 삭제
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)

    print('Training preprocessing done')

# Validation 데이터셋을 전처리하는 함수 (Training과 동일한 구조)
def preprocess_validation(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    if not source_dirs:
        raise ValueError("No directory containing 'GT' found")

    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)

    print('Validation preprocessing done')

# 데이터셋의 경로 설정
data_dir = './data/'
training_base_dir = os.path.join(data_dir, 'Training')
validation_base_dir = os.path.join(data_dir, 'Validation')

# Training과 Validation 데이터셋에 대해 전처리 수행
preprocess_training(training_base_dir)
preprocess_validation(validation_base_dir)
