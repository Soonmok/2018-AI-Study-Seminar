import pygame
import random
import math
import asyncio
from gameObject import Ball, Dead_Ball, Bonus_Ball, Cannon_Ball

pygame.init()
SCREEN_WIDTH = 700
SCREEN_HIGHT = 300

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HIGHT))
done = False
balls_num = 10
dead_balls_number = 3
Bonus_balls_number = 7

clock = pygame.time.Clock()

class GameState:
    def __init__(self):
        #initialize game objects
        self.cannon = Cannon_Ball(screen)
        self.balls = [Ball(random.randrange(1,5), screen ) for _ in range(balls_num)]
        self.dead_balls = [Dead_Ball(random.randrange(1,5), screen) for _ in range(dead_balls_number)]
        self.Bonus_Balls = [Bonus_Ball(random.randrange(1,5), screen) for _ in range(Bonus_balls_number)]

    # operation per frame
    def frame_step(self, input_actions):
        ##get mouse input
        #x, y = pygame.mouse.get_pos()
        pygame.event.pump()

        reward = -0.1
        terminal = False

        # input_actions[0] == 1 : do nothing
        # input_actions[1] == 1 : shoot cannon ball

        # if input_actions[1] == 1:    
        #     is_clicked = False
        #     self.cannon.set_pos(1, SCREEN_HIGHT)

        #while True:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    done = True
            elif input_actions[1] == 1 and not self.cannon.is_inside():
                self.cannon.set_pos(100, 100)

        for ball in self.balls:
            ball.isCollide(self.cannon)
            if ball.isCollide(self.cannon):
                reward += 0.2
            else:
                ball.draw()

            ball.reset()
            ball.update()

        for dead_ball in self.dead_balls:
            dead_ball.isCollide(self.cannon)
            if dead_ball.isCollide(self.cannon):
                reward -= -1
                terminal = True
            else:
                dead_ball.draw()

            dead_ball.reset()
            dead_ball.update()

        for bonus_ball in self.Bonus_Balls:
            bonus_ball.isCollide(self.cannon)
            if bonus_ball.isCollide(self.cannon):
                reward += 1 
            else:
                bonus_ball.draw()
            bonus_ball.reset()
            bonus_ball.update()

        self.cannon.update()
        self.cannon.draw()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        clock.tick(30)
        return image_data, reward, terminal
