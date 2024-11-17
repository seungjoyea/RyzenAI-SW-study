# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# 필요한 기본 파이썬 라이브러리 임포트
import argparse # 커맨드 라인 인자 파싱을 위한 라이브러리
import random   # 커맨드 라인 인자 파싱을 위한 라이브러리
import tarfile  # tar 파일 처리를 위한 라이브러리
import urllib.request   # URL에서 파일을 다운로드하기 위한 라이브러리

# PyTorch 관련 라이브러리 임포트
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_utils import get_directories
from torchvision.models import ResNet50_Weights, resnet50


def get_args():
    """커맨드 라인 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=0)    # 학습 에폭 수 설정
    parser.add_argument("--train", action='store_true') # 학습 모드 설정
    args = parser.parse_args()
    return args


def load_resnet_model():
    """ResNet50 모델을 로드하고 CIFAR-10용으로 마지막 레이어를 수정하는 함수"""
    # (2048 -> 64 -> 10 클래스)
    weights = ResNet50_Weights.DEFAULT  # 사전학습된 가중치 로드
    resnet = resnet50(weights=weights)  # ResNet50 모델 생성
    # 마지막 완전연결층을 CIFAR-10용으로 수정 (2048->64->10)
    resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 64),  # 2048 -> 64 차원으로 축소
        torch.nn.ReLU(inplace=True),    # ReLU 활성화 함수
        torch.nn.Linear(64, 10))    # 64 -> 10 클래스로 출력
    return resnet


# For updating learning rate
def update_lr(optimizer, lr):
    """옵티마이저의 학습률을 업데이트하는 함수"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def prepare_model(num_epochs=0, models_dir="models", data_dir="data"):
    """모델 준비 및 학습을 수행하는 메인 함수"""
    # seed everything to 0
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 설정
    num_epochs = num_epochs
    learning_rate = 0.001

    # 이미지 전처리 파이프라인 정의
    transform = transforms.Compose(
        [
            transforms.Pad(4), # 이미지 패딩
            transforms.RandomHorizontalFlip(), # 랜덤 수평 뒤집기
            transforms.RandomCrop(32), # 32x32 크기로 랜덤 크롭
            transforms.ToTensor()]  # PyTorch 텐서로 변환
    )

    # CIFAR-10 dataset 로드
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor())

    # Data loader 생성
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    # 모델 생성 및 디바이스 할당
    model = load_resnet_model().to(device)

    # Loss and optimizer 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 데이터를 디바이스로 이동
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()   # 그래디언트 초기화
            loss.backward()      # 역전파
            optimizer.step()    # 가중치 업데이트
            # 학습 상태 출력
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    if num_epochs:
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Accuracy of the model on the test images: {} %".format(100 * correct / total))

    # 모델을 CPU로 이동 후 저장
    model.to("cpu")
    torch.save(model, str(models_dir / "resnet_trained_for_cifar10.pt"))

def export_to_onnx(model, models_dir): 
    """PyTorch 모델을 ONNX 형식으로 변환하는 함수"""
    model.to("cpu") # 모델을 CPU로 이동
    dummy_inputs = torch.randn(1, 3, 32, 32)    # 더미 입력 생성
    input_names = ['input'] # ONNX 모델의 입력 이름
    output_names = ['output']   # ONNX 모델의 출력 이름
    dynamic_axes = {    # 배치 크기를 동적으로 설정
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}}
    
    # ONNX 모델 저장 경로 설정
    tmp_model_path = str(models_dir / "resnet_trained_for_cifar10.onnx")
    
    # ONNX 변환 및 저장
    torch.onnx.export(
            model,
            dummy_inputs,
            tmp_model_path,
            export_params=True,
            opset_version=13,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def main():
    """메인 함수"""
    # 필요한 디렉토리 경로 가져오기
    _, models_dir, data_dir, _ = get_directories()
    args = get_args()

    # CIFAR-10 데이터셋 다운로드
    data_download_path_python = data_dir / "cifar-10-python.tar.gz"
    data_download_path_bin = data_dir / "cifar-10-binary.tar.gz"
    # 데이터셋 URL에서 다운로드
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", data_download_path_python)
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", data_download_path_bin)
    
    # 다운로드한 파일 압축 해제
    file_python = tarfile.open(data_download_path_python)
    file_python.extractall(data_dir)
    file_python.close()
    file_bin = tarfile.open(data_download_path_bin)
    file_bin.extractall(data_dir)
    file_bin.close()

    # 학습 모드인 경우 모델 학습 수행
    if args.train:
        prepare_model(args.num_epochs, models_dir, data_dir)

    # 저장된 모델 로드 및 ONNX 변환
    model = torch.load(str(models_dir / "resnet_trained_for_cifar10.pt"))
    export_to_onnx(model, models_dir)

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()
