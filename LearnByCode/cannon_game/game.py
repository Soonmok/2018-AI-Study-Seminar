import pygame
import random
import math
import asyncio
from gameObject import Character, Barricade, Terminate
pygame.init()
screen = pygame.display.set_mode((400, 300))
done = False

# initialize objects
character = Character(screen)
barricade = Barricade(screen, 400)
terminate = Terminate(screen, 400, 300)


while not done:
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("done1")
                done = True
            
    if terminate.isCrashed(character.location_x, character.location_y):
                print("done2")
                done = True
    if barricade.iscrashed(character.location_x, character.location_y):
        print("done3")
        done = True
    pressed = pygame.key.get_pressed()
    character.update(pressed)
    barricade.update()
    screen.fill((0, 0, 0))
    character.draw()
    barricade.draw()
    terminate.draw()
    pygame.display.flip()