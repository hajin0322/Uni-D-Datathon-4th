import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Self-Attention / DownScaling
1.  Self-Attention: 입력 데이터 안에서 중요한 부분을 찾는 과정
    일반적인 Attention 과정에서는 모든 데이터를 서로 비교해 데이터 간의 상관관계를 계산
    그래서 Query, Key, Value라는 세 가지 역할로 데이터를 변환해, Query와 Key의 유사성 비교 후 Value 추출
    * Query: 우리가 집중해서 살펴보는 판단 데이터, "이 데이터가 다른 데이터와 얼마나 관련있는데?"
    * Key: 다른 데이터(단어)가 자신을 나타내는 특성, "이 데이터는 어떤 특징을 가지고 있어."
    * Value: Query와 Key의 유사도에 따른 값 반영

2.  DownScaling: 데이터 크기나 채널 수을 줄이는 것
    큰 이미지는 메모리와 계산량이 많이 들어, 이미지를 조금 작게(다운스케일) 만들어 계산 효율을 높임
'''


'''
Multi-Dconv Head Transposed Attention (MDTA) 모듈
R1280x0.png 파일 Restormer 구조 이미지에 MDTA 구조가 나와있습니다.
기존의 SA(self-attention) 같은 경우에는 각 픽셀이 하나의 token(patch)라고 가정했을 때,
하나의 이미지에서 W(Width) * H(Height) 만큼의 token들이 생깁니다.
이때 SA는 한 픽셀과 다른 모든 픽셀과의 관계를 연산하게 되는데
그러면 한 픽셀 당 연산은 W*H 번
총 한 이미지당 (W*H) * (W*H) 번 진행하게 됩니다. 즉 제곱배로 커지죠.

여기서는 해당 문제를 완화하고자 (WxH) 기준의 self-attetion을 하지 않고 channel 단위의 C x C self-attention 연산을 합니다.
즉 채널 간의 관계를 볼 수 있는 것이죠.

이미지를 구성하는 각 채널들은 서로 다른 특징을 담고 있는 데이터들이다.


그래서 attention map의 크기가 CxC가 됩니다.
'''
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        # Self-attention 온도를 학습 가능한 파라미터로 정의
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # Query, Key, Value를 생성하는 1x1 컨볼루션 레이어
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        # Q, K, V에 대한 3x3 깊이별 컨볼루션 레이어
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        # 출력 프로젝트 레이어
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 입력 텐서의 형태를 받아옴
        b, c, h, w = x.shape
        # Q, K, V를 계산하고 3개로 분할
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        # Q, K, V를 Multi-Head 형식으로 변환
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        # Q와 K를 정규화
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        # Self-attention 계산
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


'''
Gated-Dconv Feed-Forward Network (GDFN) 모듈
여기에서도 Feed-Forward의 Fully-connected layer 대신에 DConv를 사용하여 연산 효율을 높였습니다.
또한 DConv를 사용해서 GDFN 내부에서 계산할 때 채널 수를 한 번 늘렸다가 밖으로 결과를 내보낼 때 들어왔던 채널 수로 다시 맞춰줍니다.
'''
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        
        # 확장된 채널 수
        hidden_channels = int(channels * expansion_factor)
        # 입력 프로젝트 레이어
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        # Depth-wise convolution
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        # 출력 프로젝트 레이어
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x를 2개로 분할하고 Gated Mechanism 적용
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


'''
트랜스포머 블록 모듈
MDTA와 GDFN을 결합해 중요 정보 탐색 및 강화 과정을 반복하는 모듈입니다.
'''
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        # Layer Normalization과 MDTA 및 GDFN 모듈 정의
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        # MDTA와 GDFN을 잇따라 적용
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        return x


'''
Down-sampling 모듈
DownSampling은 입력을 조금 더 작고 효율적으로 만들어주는 역할을 하며, 계산량을 줄이기 위한 모듈입니다.
'''
class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        # Down-sampling 수행을 위한 컨볼루션과 PixelUnshuffle
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False), nn.Pixel)