import pygame

BALL_SIZE = 10
CANNON_BALL_SIZE = 15
COLLISION_DISTANCE = CANNON_BALL_SIZE / 2 + BALL_SIZE / 2
cannonBall_color = (255, 128, 0)
balls_color = (0, 128, 255)
dead_ball_color = (255, 0, 0)
Bonus_ball_color = (255, 255, 255)

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
        if self.location_x <= 20 or self.isCollided:
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

# 빨강     
class Dead_Ball(Ball):
    def draw(self):
        pygame.draw.circle(screen, dead_ball_color, (self.location_x, self.location_y), 5)
        
    
# 흰   
class Bonus_Ball(Ball):
    def draw(self):
        pygame.draw.ellipse(screen, Bonus_ball_color, pygame.Rect(self.location_x, self.location_y, 5, 10))
        
        
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
        self.velocity = 14 # V (고정)
        self.cursor_location = [x, y]
        self.start_point = [0, SCREEN_HIGHT] 
        self.velocity_x = self.velocity * math.cos(math.atan((SCREEN_HIGHT - y)/x)) # 초기 Vx
        self.velocity_y = self.velocity * math.sin(math.atan((SCREEN_HIGHT - y)/x)) # 초기 Vy
        
    def is_inside(self):
         return self.location_x >= 0 \
            and self.location_x <= SCREEN_WIDTH \
            and self.location_y >= 0 \
            and self.location_y <=SCREEN_HIGHT
    
 