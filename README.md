# CNN Architecture for PyTorch

CNN의 대표적인 모델인 LeNet5, AlexNet, VGG, ResNet의 구조를 살펴보고, CIFAR-10 데이터를 기준으로 PyTorch로 구현해봤습니다. 

단, 논문에서 사용한 데이터 사이즈와 CIFAR-10 데이터 사이즈가 다르므로 커널크기(kernel size)나 패딩(padding), 스트라이드(stride)가 논문에서 언급된 것과 다를 수 있습니다.

모델의 구조를 파악하기 위해 각 모델별 논문을 요약정리 해 놓은 [라온피플(주) 블로그](https://blog.naver.com/laonple/220643128255)를 주로 활용했습니다.

## Model Architectures

### LeNet5 

- **기본 구조**

  - 2개의 convolutional layers, 2개의 sub-sampling layers 및 2개의 fully-connected layers
  - convolutional layers 다음에는 Subsampling(pooling)을 적용함 (average-pooling)

  ![LeNet](https://user-images.githubusercontent.com/36662761/93849124-46e03d80-fce6-11ea-9dea-811bfe91a871.PNG)

- **특징**

  - 최초의 CNN 모델(LeNet1, 1990년)의 개선 버전
  - 가장 기본적인 CNN 모델으로 성능도 많이 떨어짐

- **References**

  -  [Gradient-Based Learning Applied to Document Recognition (Yann LeCun, 1998)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
  -  [라온피플(주) 블로그](https://blog.naver.com/laonple/220648539191)

### AlexNet

- **기본 구조** 
  - 2개의 GPU(엔비디아 GTX580)를 기반으로 한 병렬 구조 - GPU1에서는 주로 색상과 관련없는 정보를 추출하고, GPU2에서는 색상과 관련된 정보를 추출함
  - max-pooling layers가 뒤따르는 5개의 convolutional layers, 3개의 fully-conneted layers, 마지막 fully-connected layers 이후에는 1,000개의 category 분류를 위한 softmax 함수 사용
  - 65만 개의 뉴런, 6,000만 개의 파라미터, 6억 3,000만 개의 연결(connection)로 구성

![AlexNet](https://user-images.githubusercontent.com/36662761/93849129-49db2e00-fce6-11ea-9d72-6734fac6fe77.PNG)

- **세부 구조**
  - 첫 번째 convolutional layer
    - input: 224 x 224 x 3 크기의 이미지 데이터
    - kernel: 11 x 11 x 3으로 비교적 큰 receptive field를 갖는 kernel 96개 사용
    - stride: 4 
    - feature map: 55 x 55 x 96 크기
    - 뉴런 개수: 55 x 55 x 96 = 290,400개, 커널당 파라미터 개수: 364 (= 11 x 11 x 3 + 1(bias)), 총 파라미터 개수: 34,944 (= 364 x 96) - kernel 하나당 파라미터 개수는 364개인데, 그런 kernel이 96개 있으므로
    - 연결(connection): 105,750,600 (= 290,400 x 364)
  - 두 번째 convolutional layer
    - input: 첫 번째 convolutional layer의 output (ReLU와 Local Response Norm과 Max-pooling을 적용한 output)
    - kernel: 5 x 5 x 48 크기를 갖는 kernel 256개 사용
  - 세 번째 convolutional layer
    - input: 두 번째 convolutional layer의 output (ReLU와 Local Response Norm과 Max-pooling을 적용한 output)
    - kernel: 3 x 3 x 256 크기를 갖는 kernel 384개 사용
  - 네 번째 convolutional layer
    - input: 세 번째 convolutional layer의 output (ReLU만 적용한 output)
    - kernel: 3 x 3 x 192크기를 갖는 kernel 384개 사용
  - 다섯 번째 convolutional layer
    - input: 네 번째 convolutional layer의 output (ReLU만 적용한 output)
    - kernel: 3 x 3 x 192크기를 갖는 kernel 256개 사용
  - 여섯 번째/일곱 번째 fully-connected layer
    - 4,096개의 뉴런끼리 연결
  - 여덟 번째 fully-connected layer
    - 1,000개의 category와 연결
    - softmax 함수 적용
- **특징**
  - ReLU(Rectified Linear Unit)
    - 기존 모델은 비선형 활성함수(nonlinear activation function)로 sigmoid나 tanh를 사용합니다. 하지만 이 활성함수는 속도가 느립니다.
    - 이를 개선하기 위해 AlexNet은 비선형 활성함수로 ReLU를 사용합니다. ReLU는 sigmoid나 tanh에 비해  학습속도가 6배 빠릅니다.
  - Overlapped pooling
    - CNN에서 pooling을 하는 이유는 feature-map의 크기를 줄여 가장 중요한 특징을 추출하기 위함입니다. pooling에는 kernel 내에 있는 픽셀의 평균을 취하는 average-pooling이 있고, kernel 내의 픽셀 중 최대값을 선택하는 max-pooling이 있습니다. LeNet5 모델에서는 average-pooling 방식을 사용했지만 AlexNet 모델에서는 max-pooling 방식을 사용합니다.
    - 일반적으로 pooling 할 때는 겹치는 부분이 없게 합니다. 하지만 AlexNet에서는 kernel 크기보다 stride 크기를 작게하여 pooling 영역이 겹치게 했습니다. 이런 방식을 overlapped pooling이라고 하며, overlapped pooling을 적용하여 에러율을 줄였습니다. 하지만 이 방식은 과적합(overfitting)에 빠질 가능성도 있습니다.
  - Local response normalization(LRN)
    - AlexNet의 첫 번째와 두 번째 convolution을 거친 결과에 대해 ReLU를 수행하고, max-pooling을 적용하기 전에 local response normalization을 수행합니다. 
    - 이는 결과를 정규화시켜주는 효과가 있기 때문에 에러율을 더 낮출 수 있습니다.
  - Dropout
    - AlexNet은 파라미터가 6,000만 개로 상당히 많습니다. 파라미터가 이렇게 많으면 과적합(overfitting)을 일으킬 가능성이 있습니다. 이를 개선하기 위해 AlexNet은 Dropout을 적용합니다. 처음 2개의 fully-connected layers에 적용했으며, 비율은 50%로 설정했습니다.
  - Data Augmentation
    - 과적합을 개선하기 위한 또 다른 방법으로 Data Augmentation 방법이 있습니다. AlexNet에서는 2가지 Data Augmentation 방법을 적용했습니다.
    - 첫 번째로 256 x 256 크기의 원본 영상에서 224 x 224 크기의 영상을 취하는 것입니다. 그러면 1장의 영상으로부터 2,048개의 영상을 얻을 수 있습니다. 이를 훈련(training) 단계에 사용합니다. 테스트(test) 단계에서는 256 x 256 영상의 상, 하, 좌, 우 코너 및 중앙으로부터 5개의 224 x 224 영상을 추출하고 그것들을 수평으로 반전한 이미지 5개를 더해 총 10개의 영상을 활용합니다. 10개의 영상에 softmax를 적용해 평균값을 최종 결과로 사용합니다.
    - 두 번째 방법은 학습 영상의 RGB 값을 변화시키는 것입니다. 
    - Data Augmentation을 통해 1% 이상 에러율을 줄였습니다.
  - 2개의 GPU
    - 2개의 GPU(엔비디아 GTX580)를 기반으로 한 병렬 구조 - GPU1에서는 주로 색상과 관련없는 정보를 추출하고, GPU2에서는 색상과 관련된 정보를 추출함

- **References**

  - [ImageNet Classification with Deep Convolutional Neural Networks (Alex Krizhevsky, 2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
  - [라온피플(주) 블로그](https://blog.naver.com/laonple/220667260878)

### VGG Net

- **기본 구조**

  - 오직 3 x 3의 가장 간단한 kernel을 사용
  - 총 6가지의 구조로 실험 (A, A-LRN, B, C, D, E)
  - 최종 구조인 D 구조는 총 16개의 layers로 구성
  - 1억 3,300만 개의 파라미터로 구성

  ![VGG](https://user-images.githubusercontent.com/36662761/93849197-827b0780-fce6-11ea-9fdc-288916f3a885.PNG)

  <img width="740" alt="vgg_architecture2" src="https://user-images.githubusercontent.com/36662761/94353667-a9eb1f00-00ae-11eb-8b2b-ac5d3305ae01.PNG">


- **특징**
  - LRN 적용 안 함
    - A-LRN 구조는 Local Response Normalization을 적용한 구조인데, 예상과 달리 VGG Net에서는 LRN이 별로 효과가 없음
  - 기울기 소실 및 폭발 개선
    - 마지막 구조인 D구조는 layers가 16개로 굉장히 깊음. (논문에 따르면 layers가 16개 이상이면 별 이득이 없는 것으로 확인됨) 이렇게 layers가 깊을 때는 기울기 소실 및 폭발(gradient vanishing or exploding) 문제가 발생할 수 있음. 이를 개선하기 위해 비교적 간단한 A구조를 먼저 학습시킨 뒤 더 깊은 나머지 구조를 학습시킬 때, 처음 4개의 layers와 마지막 fully-connected-layers의 경우 구조 A의 학습 결과로 얻어진 값을 초기값으로 세팅함. 
  - Data Augmentation
    - AlexNet은 학습 이미지를 256 x 256 크기로 만든 후, 무작위로 224 x 224 크기의 이미지를 추출하고, RGB 컬러를 주성분 분석하여 Data Augmentation을 적용했음. 하지만 256 x 256 크기의 single scale만 활용했음
    - 반면, VGG Net은 single scale과 multi scale을 모두 활용함. single scale의 경우 256 x 256과 384 x 384 이미지를 사용함
    - multi scale은 256 x 256 ~ 512 x 512 범위에서 무작위로 scale을 정해 학습 시 사용하는 기법임. scale을 무작위로 바꿔가며 학습시킨다고 하여 이 기법을 scale jittering이라고 함.
    - 이렇게 크기를 조정한 이미지 중 무작위로 224 x 224 크기의 이미지를 선택하여 학습시 사용했으며, AlexNet과 마찬가지로 RGB 컬러를 주성분 분석하여 변화시켰음
- **References**

  -  [Very Deep Convolutional Networks For Large-scale Image Recognition(Karen Simonyan & Andrew Zisserman, 2015)](https://arxiv.org/pdf/1409.1556.pdf)
  -  [라온피플(주) 블로그](https://blog.naver.com/laonple/220738560542)

### ResNet

- **깊은 신경망(Deep Neural Network)의 문제점**

  - 기울기 소실 및 폭발(Vanishing/Exploding Gradient)
  - 성능저하(degradation): CIFAR-10 데이터를 20-layers와 56-layers 모델로 실험하여 비교했을 때, 56-layers의 에러율이 더 높게 나타남, 이는 과적합(overfitting)과 다름 - 과적합은 훈련데이터에 대한 성능은 높고 테스트데이터에 대한 성능은 높아야 하는데 아래 그리프처럼 훈련데이터와 테스트데이터 모두에 대해 성능이 하락하는 것을 볼 수 있음

  ![image-20200925214817794](https://user-images.githubusercontent.com/36662761/94275179-9bebb000-ff81-11ea-8301-fceab4e6b8a8.PNG)

- **기본 구조**

  - 기존의 CNN 구조

    - 입력 x를 전달했을 때, 출력 H(x)를 반환하는 구조

    <img width="216" alt="ResNet2" src="https://user-images.githubusercontent.com/36662761/94275191-9f7f3700-ff81-11ea-8039-fd518b6edb9f.PNG">

  - ResNet의 Residual Learning 구조

    - 깊은 신경망의 문제점인 기울기 소실/폭발과 성능저하를 개선하기 위해 제시한 구조

    - ResNet은 Residual Learning이라는 독특한 구조를 가지고 있음

    - 입력 x를 전달했을 때, 출력 H(x) - x를 반환하는 구조 [F(x) = H(x) - x라 한다면 H(x) = F(x) + x]

      <img width="449" alt="ResNet3" src="https://user-images.githubusercontent.com/36662761/94275200-a1e19100-ff81-11ea-9f5d-0df2de370f2f.PNG">

    - 입력이 바로 출력으로 연결되는 shortcut이 생긴 구조, 덧셈 연산 하나만 늘어난 것이지만 성능 향상에는 꽤 큰 효과를 가져다 줌

    - Residual Learning 개념이 고안된 이유는 VGG Net과 같은 기존 방식으로는 일정 layer 이상을 넘어가게 되면 성능이 더 나빠지는 문제를 해결하기 위함임

    - 기존의 CNN 모델과 ResNet 모델을 비교했을 때, ResNet의 성능이 더 우수함. 또한 기존 CNN 모델은 34-layers가 18-layers보다 에러율이 더 높았는데, ResNet은 34-layers가 에러율이 더 낮음

      <img width="694" alt="ResNet4" src="https://user-images.githubusercontent.com/36662761/94275214-a5751800-ff81-11ea-83fa-7b4a7fa403cf.PNG">

  - 병목 구조(Bottleneck Architecture)

    - 연산량을 줄이기 위해 3x3 convolution을 2개 연결시키는 대신 1x1 convolution을 수행하고 3x3 convolution을 수행한 뒤 다시 1x1 convolution을 수행했음

      <img width="625" alt="ResNet5" src="https://user-images.githubusercontent.com/36662761/94275221-a7d77200-ff81-11ea-81fd-a22c7b0d7ed7.PNG">
    
  - 34-layer 모델에 추가적으로 layer을 더해 50-layer, 101-layer, 152-layer도 만듦, 모두 34-layer보다 성능이 좋음, ImageNet에 최종 제출한 모델은 152-layer를 두 개 앙상블한 모델임

- **특징**
  
  - Feature-map의 크기가 절반으로 줄어드는 경우, 연산량의 균형을 맞추기 위해 kernel 수를 2배로 늘림
  - Feature-map의 크기를 줄일 때는 pooling을 사용하는 대신 convolution을 수행할 때 stride의 크기를 늘려줌 (연산량을 줄이기 위해)
- 연산량을 줄이기 위해 Dropout을 사용하지 않음
  
- **References**
  - [Deep Residual Learning for Image Recognition(Kaiming He,  2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
  - [라온피플(주) 블로그](https://blog.naver.com/laonple/220761052425)