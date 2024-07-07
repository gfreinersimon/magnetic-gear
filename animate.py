import objects
import pygame
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

def get_circle(R,n,center):
    angles = np.linspace(0,np.pi*2, n)
    xs = R*np.cos(angles)
    ys = R*np.sin(angles)+R
    return xs, ys

def animate_circle(Rs,l,n,dt,len):
    # Create a figure and axis
    fig, ax = plt.subplots()
    xs, ys = get_circle(60,l,100)
    line, = ax.plot(xs, ys)
    plt.xlim((-l,l))
    plt.ylim((-l,l))

    # Function to update the plot for each frame of the animation
    def update(frame):
        print(Rs[frame])
        xs, ys = get_circle(Rs[frame],l,n)
        line.set_xdata(xs)
        line.set_ydata(ys)  # Update y-data (e.g., changing with time)
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(len), interval=dt,repeat_delay = 1000)

    plt.show()
if __name__ == "__main__":
    l = 1
    n = 100
    Rs = np.linspace(60,0.1, 100)
    animate_circle(Rs,l,n,100,100)

output_path = 'out.csv'
dt = 0.06
omega = -2.345
center_distance = 0.095

df_s2 = pd.read_csv(output_path)
thetas = df_s2['theta[rad]'].to_numpy()
ts = df_s2['time[s]'].to_numpy()


spinner1 = objects.spinner(3,3e-2,0,np.array([0,0]))
spinner1.thetas = thetas
spinner2 = objects.spinner(3,3e-2,0,np.array([center_distance,0]))
spinner2.thetas = ts * omega

animate(spinner1,spinner2,dt,1600,800,5000.)
