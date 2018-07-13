import numpy as np
from image_processing import process_image, stack_images

def init_env_data(game_state):
    # 초기 상태를 점프뛰지 않는 상태로 두고 이미지를 80 * 80 * 4 형태로 preprocessing함
    """ do_nothing == [1, 0]"""
    do_nothing = np.zeros(2)
    do_nothing[0] = 1

    """ game_state.frame_step => 행동 input을 기반으로 게임에 변화를 줌
    x_t == 이미지 데이터, r_0 == reward, terminal == 게임이 끝났는지 여부"""
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    images = process_image(x_t)
    s_t = stack_images(images)
    return s_t

def update_env_by_action(game_state, s_t, a_t):
    # 계산된 행동을 하고 그 행동에 따른 게임 상태를 받아오고 데이터 처리함

    """ x_t1_colored = 게임 이미지(컬러), 
        r_t = 행동에 따른 보상값,
        terminal = 끝난 여부
    """
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

    image = process_image(x_t1_colored)
    image = np.reshape(image, (80, 80, 1))
    s_t1 = np.append(image, s_t[:, :, :3], axis=2)
    return s_t1, r_t, terminal
