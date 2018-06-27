# Using Deep Q-Network to Learn How To Play Flappy Bird

<img src="./images/flappy_bird_demp.gif" width="250">

7 mins version: [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)

## Overview
이 프로젝트는 딥 마인드의 강화학습을 이용한 아타리 게임 플레이하기 프로젝트의 알고리즘을 따와서 flappy bird 라는 게임에 적용시킨 프로젝트입니다.

## Installation Dependencies:
* Python 2.7 or 3
* TensorFlow 0.7
* pygame
* OpenCV-Python

사용하기 쉽도록 docker image를 만들어 곧 베포할 예정

## 실행 방법
```
cd DeepLearningFlappyBird
python deep_q_network.py
```

## What is Deep Q-Network?
DQN(Deep Q Network)란?

convolution neural network와 q learning 을 이용하여 cnn으로 픽셀데이터를 읽고 특징을 읽고 q-learning으로 value function (강화학습을 시키기 위해 보상체계)를 구현하는 것

## Deep Q-Network Algorithm

pseudo-code

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```

## Experiments

#### Environment

q network가 읽어온 픽셀을 기반으로 학습을 하기 때문에 불필요한 배경 정보는 빼도록 아래와 같이 게임 환경을 바꿈

<img src="./images/preprocess.png" width="450">

#### Network Architecture
학습에 필요한 이미지 데이터를 처리하는 과정

1. 이미지를 흑백으로 바꾼다.
2. 이미지를 80 * 80 크기로 재설정한다.
3. 4개의 frame들을 쌓아서 80 * 80 * 4의 형태를 갖춘 배열을 만들고 이 배열을 인풋데이터로 사용한다.

아래 그림에 자세한 아키텍처가 설명되어 있습니다.
간단히 말하자면 앞서 말한 input 데이터에 convolution과 maxpooling 을 총 3번 시키고 아웃풋을 0 과 1 (점프를 뛴다 = 1 , 점프를 뛰지 않는다 = 0) 으로 나오도록 한다. 


<img src="./images/network.png">

각각의 timestep마다 학습하는 모델은 더 높은 q value값을 얻기위해 노력하면서 학습한다.


## Disclaimer
이 프로젝트는 아래에 있는 깃허브 프로젝트를 따온 것입니다.

1. (https://github.com/yenchenlin/DeepLearningFlappyBird)

