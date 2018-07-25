# 게임 이미지데이터를 처리하기 위한 opencv 라이브러리 입니다.
import cv2
import numpy as np

def process_image(image):
    """ 이미지를 흑백으로, 80 * 80 크기로 잘라냄"""
    image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)

    """ 이미지 임계처리"""
    ret, image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    return image

def stack_images(image):
    """ 80 * 80 * 4 형태로 만듬
    80 * 80 크기의 이미지 4쌍이 한 세트""" 
    state_data = np.stack((image, image, image, image), axis=2)
    return state_data
