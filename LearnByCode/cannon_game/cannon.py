import pygame
import random
import math
import asyncio


pygame.init()
SCREEN_WIDTH = 700
SCREEN_HIGHT = 300
BALL_SIZE = 10
CANNON_BALL_SIZE = 15
COLLISION_DISTANCE = CANNON_BALL_SIZE / 2 + BALL_SIZE / 2
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HIGHT))
done = False
balls_num = 10
dead_balls_number = 3
Bonus_balls_number = 2
cannonBall_color = (255, 128, 0)
balls_color = (0, 128, 255)
dead_ball_color = (255, 0, 0)
Bonus_ball_color = (255, 255, 255)
clock = pygame.time.Clock()


class Ball:
    def __init__(self, velocity):
        self.location_x = SCREEN_WIDTH + 100 + random.randrange(100)
        self.location_y = random.randrange(300)
        self.velocity = velocity
        self.isCollided = False
        
    def draw(self):
        pygame.draw.rect(screen, balls_color, pygame.Rect(self.location_x, self.location_y, BALL_SIZE, BALL_SIZE))
        
    def update(self):
        self.location_x -= self.velocity
        
    def reset(self):
        
        if self.location_x <= 20 or self.isCollided :
            
            self.location_x = SCREEN_WIDTH + 100 + random.randrange(100)
            self.isCollided = False
            
            
    def isCollide(self, cannon_ball):
        if math.sqrt(math.pow((cannon_ball.location_x - self.location_x), 2) \
                  + math.pow((cannon_ball.location_y - self.location_y), 2)) < COLLISION_DISTANCE:
            self.isCollided = True
            return True
        else:
            self.isCollided = False
            return False

class Dead_Ball(Ball):
    def draw(self):
        pygame.draw.rect(screen, dead_ball_color, pygame.Rect(self.location_x, self.location_y, BALL_SIZE, BALL_SIZE))
        
    
class Bonus_Ball(Ball):
    def draw(self):
        pygame.draw.rect(screen, Bonus_ball_color, pygame.Rect(self.location_x, self.location_y, BALL_SIZE, BALL_SIZE))
        
class Cannon_Ball:
    def __init__(self):
        self.location_x = -1
        self.location_y = -1
    
    def draw(self):
        pygame.draw.rect(screen, cannonBall_color, pygame.Rect(self.location_x, self.location_y, CANNON_BALL_SIZE, CANNON_BALL_SIZE))
        
    def update(self):
        self.location_x += self.velocity_x
        self.location_y -= self.velocity_y 
        self.velocity_y -= 0.35
    
    def set_pos(self, x, y):
        self.location_x = 0
        self.location_y = SCREEN_HIGHT
        self.velocity = 14
        self.cursor_location = [x, y]
        self.start_point = [0, SCREEN_HIGHT]
        self.velocity_x = self.velocity * math.cos(math.atan((SCREEN_HIGHT - y)/x))
        self.velocity_y = self.velocity * math.sin(math.atan((SCREEN_HIGHT - y)/x))
        
    def is_inside(self):
        if self.location_x >= 0 and self.location_x <= SCREEN_WIDTH \
        and self.location_y >= 0 and self.location_y <=SCREEN_HIGHT:
            return True
        else:
            return False
    
        
balls = [Ball(random.randrange(1,5) ) for _ in range(balls_num)]
dead_balls = [Dead_Ball(random.randrange(1,5) ) for _ in range(dead_balls_number)]
Bonus_Balls = [Bonus_Ball(random.randrange(1,5) ) for _ in range(Bonus_balls_number)]

x, y = pygame.mouse.get_pos()

cannon = Cannon_Ball()
is_clicked = False
cannon.set_pos(1, SCREEN_HIGHT)
while True:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
                done = True
        elif event.type == pygame.MOUSEBUTTONUP and not cannon.is_inside():
            
            pos_x, pos_y = pygame.mouse.get_pos()
            cannon.set_pos(pos_x, pos_y)
    
    for ball in balls:
        ball.isCollide(cannon)
        if not ball.isCollide(cannon):
            ball.draw()
        ball.reset()
        ball.update()
    
    for dead_ball in dead_balls:
        dead_ball.isCollide(cannon)
        if not dead_ball.isCollide(cannon):
            dead_ball.draw()
        dead_ball.reset()
        dead_ball.update()

    for bonus_ball in Bonus_Balls:
        bonus_ball.isCollide(cannon)
        if not bonus_ball.isCollide(cannon):
            bonus_ball.draw()
        bonus_ball.reset()
        bonus_ball.update()
    
    cannon.update()
    cannon.draw()
    
    
    pygame.display.flip()

    clock.tick(60)
