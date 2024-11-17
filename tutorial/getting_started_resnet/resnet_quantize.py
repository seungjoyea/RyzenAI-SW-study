
# 필요한 라이브러리 임포트
import torch    # PyTorch 딥러닝 프레임워크
from onnxruntime.quantization.calibrate import CalibrationDataReader    # 양자화를 위한 캘리브레이션 도구
from torch.utils.data import DataLoader, Dataset     # 데이터 로딩 유틸리티
from torchvision import transforms  # 이미지 변환 도구
from torchvision.datasets import CIFAR10    # CIFAR-10 데이터셋

# ONNX 관련 라이브러리 임포트
import onnx # ONNX 모델 처리
import onnxruntime  # ONNX 모델 실행
from onnxruntime.quantization import (
    CalibrationDataReader, # 캘리브레이션 데이터 읽기
    QuantType, # 양자화 타입 정의
    QuantFormat, # 양자화 포맷 정의
    CalibrationMethod, # 캘리브레이션 방법
    quantize_static # 정적 양자화 함수
)

import vai_q_onnx   # AMD/Xilinx AI 양자화 도구


class CIFAR10DataSet:
    """CIFAR-10 데이터셋을 로드하고 전처리하는 클래스"""
    def __init__(
        self,
        data_dir,   # 데이터 디렉토리 경로
        **kwargs,   # 추가 파라미터
    ):
        super().__init__()
        self.train_path = data_dir  # 학습 데이터 경로
        self.vld_path = data_dir    # 검증 데이터 경로
        self.setup("fit")   # 데이터셋 초기 설정

    def setup(self, stage: str):
        """데이터셋 설정 및 전처리 파이프라인 정의"""
        # 이미지 전처리 변환 정의
        transform = transforms.Compose([
            transforms.Pad(4),  # 4픽셀 패딩
            transforms.RandomHorizontalFlip(),  # 랜덤 수평 뒤집기
            transforms.RandomCrop(32),  # 32x32 크기로 랜덤 크롭
            transforms.ToTensor()]  # 텐서로 변환
        )
        # 학습용과 검증용 데이터셋 생성
        self.train_dataset = CIFAR10(root=self.train_path, train=True, transform=transform, download=False)
        self.val_dataset = CIFAR10(root=self.vld_path, train=True, transform=transform, download=False)


class PytorchResNetDataset(Dataset):
    """PyTorch 데이터셋 형식으로 CIFAR-10 데이터를 변환하는 래퍼 클래스"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        """데이터셋의 크기 반환"""
        return len(self.dataset)

    def __getitem__(self, index):
        """특정 인덱스의 데이터 샘플 반환"""
        sample = self.dataset[index]
        input_data = sample[0]  # 이미지 데이터
        label = sample[1]   # 레이블
        return input_data, label


def create_dataloader(data_dir, batch_size):
    """데이터로더 생성 함수"""
    # CIFAR-10 데이터셋 인스턴스 생성
    cifar10_dataset = CIFAR10DataSet(data_dir)
    # 검증 데이터셋을 49000:1000으로 분할
    _, val_set = torch.utils.data.random_split(cifar10_dataset.val_dataset, [49000, 1000])
    # 데이터로더 생성 및 반환
    benchmark_dataloader = DataLoader(PytorchResNetDataset(val_set), batch_size=batch_size, drop_last=True)
    return benchmark_dataloader


class ResnetCalibrationDataReader(CalibrationDataReader):
    """ResNet 모델 양자화를 위한 캘리브레이션 데이터 reader"""
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        # 데이터로더 이터레이터 생성
        self.iterator = iter(create_dataloader(data_dir, batch_size))

    def get_next(self) -> dict:
        """다음 배치의 캘리브레이션 데이터 반환"""
        try:
            images, labels = next(self.iterator)
            return {"input": images.numpy()}    # numpy 배열로 변환하여 반환
        except Exception:
            return None


def resnet_calibration_reader(data_dir, batch_size=16):
    """캘리브레이션 데이터 리더 생성 함수"""
    return ResnetCalibrationDataReader(data_dir, batch_size=batch_size)



def main():
    """메인 함수: 모델 양자화 실행"""
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = "models/resnet_trained_for_cifar10.onnx"

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = "models/resnet.qdq.U8S8.onnx"

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = "data/"

    # `dr` (Data Reader) is an instance of ResNetDataReader, which is a utility class that 
    # reads the calibration dataset and prepares it for the quantization process.
    dr = resnet_calibration_reader(calibration_dataset_path)

    # `quantize_static` is a function that applies static quantization to the model.
    # The parameters of this function are:
    # - `input_model_path`: the path to the original, unquantized model.
    # - `output_model_path`: the path where the quantized model will be saved.
    # - `dr`: an instance of a data reader utility, which provides data for model calibration.
    # - `quant_format`: the format of quantization operators. Need to set to QDQ or QOperator.
    # - `activation_type`: the data type of activation tensors after quantization. In this case, it's QUInt8 (Quantized Int 8).
    # - `weight_type`: the data type of weight tensors after quantization. In this case, it's QInt8 (Quantized Int 8).
    # - `enable_dpu`: (Boolean) determines whether to generate a quantized model that is suitable for the DPU. If set to True, the quantization process will create a model that is optimized for DPU computations.
    # - `extra_options`: (Dict or None) Dictionary of additional options that can be passed to the quantization process. In this example, ``ActivationSymmetric`` is set to True i.e., calibration data for activations is symmetrized. 
    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        enable_dpu=True, 
        extra_options={'ActivationSymmetric': True} 
    )
    print('Calibrated and quantized model saved at:', output_model_path)

if __name__ == '__main__':
    main()



#################################################################################  
#License
#Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
