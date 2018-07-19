# 학습을 위한 툴인 텐서플로우를 불러옵니다
import tensorflow as tf

ACTIONS = 2

def weight_variable(shape):
    """ 
        인공신경망 레이어에 들어가는 가중치 변수 초기화 및 반환
        
        파라메터 
        shape == 설정할 배열의 형태를 가지고 있음 ex) (3,2)

    """

    # ... 을 채우시오
    # hint 
    # tensorflow 에서 제공하는 tf.truncated_normal를 통해 랜덤 변수값(initial)을 생성할것
    """
    initial = ...
    return tf.Variable(initial)
    """
    return

def bias_variable(shape):
    """ 
        인공신경망 레이어에 들어가는 편차 변수 초기화 및 반환
        
        파라메터 
        shape == 배열의 형태를 가지고 있음 ex) (3, 2)

    """

    # ... 을 채우시오
    # tensorflow 에서 제공하는 tf.truncated_normal를 통해 랜덤 변수값(initial)을 생성할것
    """
    initial = ...
    return tf.Variable(initial)
    """
    return

def conv2d(x, W, stride):
    """
        convolution layer 생성

        파라메터

        인풋 데이터 == x,
        레이어의 weight(가중치) == w
        stride == convolution layer에 쓰이는 stride의 가로, 세로 길이(정사각형)
        
    """
    # ...을 채우시오
    # tensorflow 에서 제공하는 tf.nn.conv2d 함수를 통하여 레이어를 생성할것
    """
    return ...
    """
    return 

def max_pool_2x2(x):
    """
        convolution layer에 필요한 pooling layer 생성

        파라메터

        인풋 데이터 == x
    """

    # ...을 채우시오
    # tensorflow에서 제공하는 tf.nn.max_pool 함수를 사용하여 max pooling 레이어를 생성할것
    # 커널 사이즈는 [2, 2], stride 사이즈도 [2,2]로 하고 padding을 해야함
    """
    return ...
    """
    return

def create_cnn_layer(data, weight, bias, stride):
    """
        앞서서 만든 weight_variable 함수와, bias_variable 함수를 이용하여
        convolution layer를 만들어 반환

        파라메터 
        data = 인풋 데이터
        weight = 가중치 값
        bias = 편향 값
        stride = stride 의 가로세로 길이
        
    """
    
    # ...을 채우시오
    # 앞서서 만든 함수를 사용하여 weight, bias 생성
    # tensorflow에서 제공하는 tf.nn.relu함수 사용
    # 위에서 만든 conv2d함수 사용
    """
    weight = ...
    bias = ...

    return ...
    """

    return 

def create_fc_layer(data, weight, bias):
    """
        fully connected layer 생성

        파라메터 
        data = 인풋 데이터
        weight = 가중치의 shape
        bias = 편향의 shape
    """

    # 앞서서 만든 함수를 사용하여 weight, bias 생성
    # tensorflow에서 제공하는 tf.nn.relu함수 사용
    """
    weight = ...
    bias = ...

    return ...
    """
    return

def createNetwork():
    """
        앞서 만들었던 함수들을 이용하여 게임 데이터(사진)을 통과시켜
        0 == 아무것도 안하기
        1 == 점프뛰기 
        를 결정하는 인공 신경망을 만드는 함수"""

    # ...을 채우시오

    # tensorflow에서 제공하는 tf.placeholder 함수사용
    """ 데이터 타입 == float,

        데이터 형태 == 80 * 80크기의 이미지 데이터가 
        4개씩 한 set로
        정해지지 않은 갯수만큼 저장될 수 있어야 함"""
    
    """
    data = ...
    """

    # 앞에서 만든 create_cnn_layer 사용
    """ weight 형태 = [8, 8, 4, 32]
        bias 형태 = [32]
        stride 가로세로 크기 = 4"""
    """
    h_conv1 = ...
    """
    
    # 앞에서 만든 max_pool_2x2 함수를 이용하여 pooling layer 생성 및 데이터통과
    """ 데이터 == h_conv1"""

    """
    h_pool1 = ...
    """

    # 앞에서 만든 create_cnn_layer 함수를 통하여 두번째 cnn 레이어를 생성 및 데이터통과
    """ 데이터 == h_pool1
        weight 형태 == [4, 4, 32, 64]
        bias == [64]
        stride 가로 세로 길이 == 2"""

    """
    h_conv2 = ...
    """

    # 앞에서 만든 create_cnn_layer 함수를 통하여 세번째 cnn 레이어를 생성
    """ 데이터 == h_conv2
        weight 형태 == [3, 3, 64, 64]
        bias 형태 == [64]
        stride 가로 세로 길이 == 1"""

    """
    h_conv3 = ...
    """

    # tensorflow에서 제공하는 tf.reshape함수를 사용하여 레이어들을 통과한 데이터들의
    # 형태를 [-1, 1600]으로 바꿈 (일자로 쭉 핌)
    """
        데이터 == h_conv3"""
    """
    h_conv3_flat = ...
    """

    # 앞에서 만든 create_fc_layer 함수를 사용하여 fully connected 레이어 생성및 통과
    """ 데이터 == h_conv_flat
        weight 형태 == [1600, 512]
        bias 형태 == [512]"""
    """
    h_fc1 = ...
    """

    # 앞에서 만든 create_fc_layer 함수를 사용하여 fully connected 레이어 생성 및 통과
    # 데이터를 통과시키면 [2, 1] 형태의 데이터가 남게되며, 
    # 이 데이터가 [1, 0]이면 가만히 있고, [0,1]이면 점프를 뛰어야 한다고 예측한다
    """
    readout = create_fc_layer(h_fc1, weight=[512, ACTIONS], bias=[ACTIONS])
    """

    return data, readout, h_fc1, h_conv1, h_pool1, h_conv2, h_conv3, h_conv3_flat
