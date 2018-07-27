import pygame
import random
import math

BALL_SIZE = 20
CANNON_BALL_SIZE = 15
COLLISION_DISTANCE = math.sqrt(2) * BALL_SIZE
cannonBall_color = (255, 128, 0)
balls_color = (0, 128, 255)
dead_ball_color = (255, 0, 0)
Bonus_ball_color = (255, 255, 255)
SCREEN_WIDTH = 700
SCREEN_HIGHT = 300
INITIAL_VELOCITY = 14


# Basic Ball 
class Ball:
    def __init__(self, screen, location_x, location_y):
        # Ball의 좌표 값, screen 설정
        self.location_x = location_x
        self.location_y = location_y
        self.screen = screen

        # 충돌여부 초기화
        self.isCollided = False

    # 사각형 
    def draw(self):
        pygame.draw.rect(self.screen, balls_color, pygame.Rect(self.location_x, self.location_y, BALL_SIZE, BALL_SIZE))

    # 사각형의 x 좌표를 변화 시킨다. 
    def update(self):
        pass #20180727

    # 위치가 화면을 벗어나거나, 공이 충돌한 경우 원래 위치로 초기화 시켜준다.
    def reset(self):
        pass #20180727
        if self.location_x <= 20 or self.isCollided:
            self.location_x = SCREEN_WIDTH + 100 + random.randrange(100)
            self.isCollided = False
    
    # cannon_ball 과 Ball 사이의 거리가 COLLISION_DISTANCE 보다 작다면 충돌했다
    # 가정.
    def isCollide(self, cannon_ball):
        if math.sqrt(math.pow((cannon_ball.location_x - self.location_x), 2) \
                  + math.pow((cannon_ball.location_y - self.location_y), 2)) < COLLISION_DISTANCE:
            self.isCollided = True
            return True
        else:
            self.isCollided = False
            return False

# 빨강 Ball     
class Dead_Ball(Ball):
    def draw(self):
        pygame.draw.circle(self.screen, dead_ball_color, (self.location_x, self.location_y), 20)

# 흰 Ball  
class Bonus_Ball(Ball):
    def draw(self):
        pygame.draw.ellipse(self.screen, Bonus_ball_color,
                            pygame.Rect(self.location_x, self.location_y, 20, 20))

# Cannon_ball
class Cannon_Ball:
    def __init__(self, screen):
        self.isFlying = False
        self.location_x = -1
        self.location_y = SCREEN_HIGHT
        self.velocity_x = 1
        self.velocity_y = 1
        self.screen = screen
        self.velocity = INITIAL_VELOCITY # V (고정)

    def draw(self):
        pygame.draw.rect(self.screen, cannonBall_color,
                         pygame.Rect(self.location_x, self.location_y, CANNON_BALL_SIZE, CANNON_BALL_SIZE))

    def update(self):
        self.location_x += self.velocity_x
        self.location_y -= self.velocity_y
        self.velocity_y -= 0.35 #0.35 means G

    def set_pos(self, input_x, input_y):
        self.location_x = 0
        self.location_y = SCREEN_HIGHT
        #self.start_point = [0, SCREEN_HIGHT] #20180725
        self.velocity_x = self.velocity *\
            math.cos(math.atan((SCREEN_HIGHT - input_y)/input_x)) # 초기 Vx
        self.velocity_y = self.velocity *\
            math.sin(math.atan((SCREEN_HIGHT - input_y)/input_x)) # 초기 Vy

    def reset(self):
        self.location_x = -1
        self.location_y = SCREEN_HIGHT

    def is_inside(self):
         return self.location_x >= 0 \
            and self.location_x <= SCREEN_WIDTH \
            and self.location_y >= 0 \
            and self.location_y <=SCREEN_HIGHT
