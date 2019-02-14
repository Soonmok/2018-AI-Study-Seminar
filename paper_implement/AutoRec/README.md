## AutoRec

### AutoEncoder 를 사용하여 Collaborating Filterring을 구현

### 논문 제목 : AutoRec: Autoencoders Meet Collaborative Filtering

Requirements 

`nvidia-docker`

or 

`tensorflow-gpu`
`pandas`
`scikit-learn`
`tqdm`

docker image name = soonmok/autorec:latest

docker pull soonmok/autorec:latest


실행방법 (nvidia docker 이용)

```docker pull soonmok/autorec:latest

git clone https://github.com/Soonmok/2018-AI-Study-Seminar.git

cd 2018-AI-Study-Seminar/paper_implement/AutoRec

wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip (작은 데이터셋 1MB)

wget http://files.grouplens.org/datasets/movielens/ml-latest.zip (큰 데이터셋 약 200MB)

docker run -it --runtime=nvidia -v $PWD:/app soonmok/autorec:latest bash (작은 데이터셋)

cd /app

python main.py --data_path=./ml-1m/ratings.dat

docker run -it --runtime=nvidia -v $PWD:/app soonmok/autorec:latest bash (큰 데이터셋)

cd /app 

python main.py --data_path=./ml-latest/ratings.csv
```



실행방법 (local 환경세팅)

```pip install requirements.txt

 python main.py --data_path=./ml-1m/ratings.dat
