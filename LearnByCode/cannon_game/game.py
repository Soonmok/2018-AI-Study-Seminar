import pygame
import random
import math
import asyncio
from gameObject import Character, Barricade, Terminate
pygame.init()
screen = pygame.display.set_mode((400, 300))
done = False
FPSCLOCK = pygame.time.Clock()



class GameState:
    def __init__(self):
        # initialize objects
        self.character = Character(screen)
        self.barricade = Barricade(screen, 300, 0)
        self.barricade2 = Barricade(screen, 200, 200)
        self.terminate = Terminate(screen, 400, 300)

    def frame_step(self, input_actions):
        # internally process pygame event handlers
        pygame.event.pump()

        reward = -0.1
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
        # check if character collides with barricade
        

        if self.barricade.iscrashed(self.character.location_x, self.character.location_y) or self.barricade2.iscrashed(self.character.location_x, self.character.location_y):
            reward = -1
            print("crashed")
            terminal = True
            self.__init__()
        if self.terminate.isCrashed(self.character.location_x, self.character.location_y):
            reward = 1
            print("crashed terminal")
            terminal = True
        screen.fill((0, 0, 0))
        self.character.draw()
        self.barricade.draw()
        self.barricade2.draw()
        self.terminate.draw()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(30)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal
