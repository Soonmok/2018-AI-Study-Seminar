import pygame
import random
import math

pygame.init()
SCREEN_WIDTH = 400
SCREEN_HIGHT = 300
BALL_SIZE = 10
CANNON_BALL_SIZE = 10
COLLISION_DISTANCE = CANNON_BALL_SIZE / 2 + BALL_SIZE / 2
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HIGHT))
done = False
balls_num = 10
cannonBall_color = (255, 128, 0)
balls_color = (0, 128, 255)
clock = pygame.time.Clock()


class Ball:
    def __init__(self, velocity):
        self.location_x = random.randrange(200, 380)
        self.location_y = random.randrange(280)
        self.velocity = velocity
        
    def draw(self):
        pygame.draw.rect(screen, balls_color, pygame.Rect(self.location_x, self.location_y, BALL_SIZE, BALL_SIZE))
        
    def update(self):
        self.location_x -= self.velocity
        
    def reset(self):
        if self.location_x <= 20:
            self.location_x = random.randrange(250, 380)
    
    def isCollide(self, cannon_ball):
        if math.sqrt(math.pow((cannon_ball.location_x - self.location_x), 2) \
                  + math.pow((cannon_ball.location_y - self.location_y), 2)) < COLLISION_DISTANCE:
            return True
        else:
            return False

class Cannon_Ball:
    def __init__(self):
        self.location_x = -1
        self.location_y = -1
    
    def draw(self):
        pygame.draw.rect(screen, cannonBall_color, pygame.Rect(self.location_x, self.location_y, CANNON_BALL_SIZE, CANNON_BALL_SIZE))
        
    def update(self):
        self.location_x += self.velocity_x
        self.location_y -= self.velocity_y 
        self.velocity_y -= 0.1
    
    def set_pos(self, x, y):
        self.location_x = 0
        self.location_y = 280
        self.velocity = 7
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
    
        
balls = [Ball(random.randrange(1,3) ) for _ in range(balls_num)]
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
    
    dup = balls.copy()
    balls.clear()
    
    for ball in dup :
        if not ball.isCollide(cannon) :
            balls.append(ball)
    
    for ball in balls:
        ball.reset()
        ball.draw()
        ball.update()


    cannon.update()
    cannon.draw()
    
    
    pygame.display.flip()

    clock.tick(60)
