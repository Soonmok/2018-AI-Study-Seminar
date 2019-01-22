## 프로젝트 설명

두가지의 에이전트 학습 방법으로 나뉘어져 있으며, 기본적인 DQN 방식과 A3C 방식으로 나뉨

먼저 모델학습을 gui 환경 없이 바로 atari API를 이용하여 빠르게 학습을 시킨 후

학습된 모델을 play_##_model.py 파일에서 불러오는 방식

## 실행 환경 dependency
```
tensorflow-gpu (or tensorflow)
Cmake
gym[atari]
```
## dependency 로컬환경 설치방법

```
pip install tensorflow-gpu (or pip install tensorflow)
sudo apt-get install cmake
pip install gym[atari]
```

## 모두를 행복하게 하는 docker를 활용한 dependency 해결방법

gpu용 docker image 이기 때문에 gpu가 없을시 돌지 않습니다.

또한 nvidia-docker가 아닌 그냥 docker를 사용할 경우 돌지 않습니다.

에러문에 lib.so .. 이런게 뜰경우 100퍼 gpu문제

```
docker pull soonmok/2018-ai-study-seminar
docker run -it --runtime=nvidia soonmok/2018-ai-study-seminar bash
```
## 실행방법

### 1. DQN

```
cd 2018-AI-Study-Seminar/LearnByOthersCode/Reinforcement_learning/Breakout
python breakout_dqn.py
학습을 마친뒤
python play_dqn_model.py
```

### 2. A3C

```
cd 2018-AI-Study-Seminar/LearnByOthersCode/Reinforcement_learning/Breakout
python breakout_a3c.py
학습을 마친뒤
python play_a3c_model.py
```
