import objects
import pygame
import sys
import numpy as np

def animate(spinner1,spinner2, dt, width, height,scale):
    # Initialize Pygame
    pygame.init()

    # Set up the screen dimensions
    screen_width = width
    screen_height = height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Draw Circle")

    # Set up colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)



    # Set up the circle parameters
    circle_radius = 50
    circle_position = (screen_width // 2, screen_height // 2)

    # Main loop
    running = True
    n = 0
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the screen with white color
        screen.fill(WHITE)
        center = np.array([width/2, height/2])

        # Draw the circle
        for i in range(spinner1.n_arms):
            pygame.draw.circle(screen, RED, np.round(np.array([1,-1])*spinner1.get_nthmi(n,i)*scale+center), scale*spinner1.R_magnets/5.)
        for i in range(spinner2.n_arms):
            pygame.draw.circle(screen, BLUE, np.round(np.array([1,-1])*spinner2.get_nthmi(n,i)*scale+center), scale*spinner2.R_magnets/5.)
        n = n+1
        pygame.time.delay(int(1000*dt))

        # Update the display
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()
    sys.exit()
