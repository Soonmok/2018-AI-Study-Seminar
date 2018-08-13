import pygame
import random
import math
import asyncio
from gameObject import Character, Barricade, Terminate

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
done = False
FPSCLOCK = pygame.time.Clock()

def get_distance(location1, location2):
    x = (location2[0] - location1[0]) * (location2[0] - location1[0])
    y = (location2[1] - location1[1]) * (location2[1] - location1[1])
    return math.sqrt(x + y)

class GameState:
    epoch_count = 0
    def __init__(self):
        # initialize objects
        self.epoch_count += 1
        self.character = Character(screen)
        self.barricade = Barricade(screen, random.randrange(0,300), 100, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.barricade2 = Barricade(screen, random.randrange(0,300), 200, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.terminate = Terminate(screen, random.randrange(0,300), SCREEN_WIDTH, SCREEN_HEIGHT)
        self.timer = 0

    def frame_step(self, input_actions):
        # internally process pygame event handlers
        pygame.event.pump()

        reward = 0
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: stay
        # input_actions[1] == 1: go left
        # input_actions[2] == 1: go right
        # input_actions[3] == 1: go up
        # input_actions[4] == 1: go down

        else:
            self.character.update(input_actions)

        self.barricade.update()
        self.barricade2.update()
        self.terminate.update()
        reward = -(get_distance(self.character.location, self.terminate.location)/ 100000)
        
        if self.barricade.iscrashed(self.character.location_x, self.character.location_y)\
        or self.barricade2.iscrashed(self.character.location_x, self.character.location_y):
            reward = -1
            print("crashed")
            terminal = True
            
        if self.terminate.isCrashed(self.character.location_x, self.character.location_y):
            reward = 1000
            print("crashed terminal")
            terminal = True
            
        if not (self.timer % 100):
            if self.timer == 1000:
                self.timer = 0
                terminal = True
            else :
                print("================================================================")
                print("Game Timer : %d" % (self.timer))


        screen.fill((0, 0, 0))
        self.character.draw()
        self.barricade.draw()
        self.barricade2.draw()
        self.terminate.draw()
        
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        if terminal:
            print("=================================================================")
            print("Epoch count : %d" %  (self.epoch_count))
            self.__init__()

        pygame.display.update()
        FPSCLOCK.tick(30)
        self.timer += 1
        
        return image_data, reward, terminal
