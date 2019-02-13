## AutoRec

### AutoEncoder 를 사용하여 Collaborating Filterring을 구현

### 논문 제목 : AutoRec: Autoencoders Meet Collaborative Filtering

Requirements 

`nvidia-docker`

docker image name = soonmok/autorec:small_data
docker image name = soonmok/autorec:big_data

실행방법 (nvidia docker 이용)

```docker pull soonmok/autorec:latest

git clone https://github.com/Soonmok/2018-AI-Study-Seminar.git

cd 2018-AI-Study-Seminar/paper_implement/AutoRec

wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip     (작은 데이터셋 1MB)

wget http://files.grouplens.org/datasets/movielens/ml-latest.zip (큰 데이터셋 약 200MB)

docker run -it --runtime=nvidia -v $PWD:/app soonmok/autorec:small_data     (작은 데이터셋)

docker run -it --runtime=nvidia -v $PWD:/app soonmok/autorec:big_data      (큰 데이터셋)
```



실행방법 (local 환경세팅)

```pip install requirements.txt

 python main.py --data_path=./ml-1m/ratings.dat
