import numpy as np
import sys
import pygame
import argparse
from robot.robot import Robot


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='2D Robotic arm inverse kinematics.')

    parser.add_argument('-m', '--method', dest='method', default='n', required=False, choices='ptn',
                        help='Method to calculate inverse kinematics: p->pseude transpose, t->transpose, n->null space')
    args = parser.parse_args()

    # Init pygame
    pygame.init()
    # Create screen
    size = width, height = 640, 480
    pygame.display.set_caption('2D redundant robotic arm')
    screen = pygame.display.set_mode(size)
    # Define colors
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    # Robot
    # Define robot origin
    origin = np.array([0.0, 0.0])
    # Load Robot object
    robot = Robot(origin)
    # Define robot initial position
    q_pos = np.matrix([[0.1], [0.3], [0.2]])
    # Defined robot default position for the null-state solver
    q_star = np.matrix([[np.pi/2.0], [0], [np.pi/2.0]])

    # Start infinite loop
    while True:

        # FPS control
        start_time = pygame.time.get_ticks()

        # Events handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Capture mouse position
        x_mouse, y_mouse = pygame.mouse.get_pos()

        # Transform mouse position to robot coordinates frame
        x_mouse = x_mouse - width/2.0
        y_mouse = height/2.0 - y_mouse

        # Compute robot inverse kinematics
        if args.method == 'p':
            q_pos = robot.inverse_kinematics_pseudo_inverse(
                np.matrix([[x_mouse], [y_mouse]]), q_pos)
        elif args.method == 't':
            q_pos = robot.inverse_kinematics_transpose(
                np.matrix([[x_mouse], [y_mouse]]), q_pos)
        else:
            q_pos = robot.inverse_kinematics_null_space(
                np.matrix([[x_mouse], [y_mouse]]), q_pos, q_star)

        # Compute robot forward kinematics
        x_pos = robot.forward_kinematics(q_pos)

        # Transform robot position to pygame coordinates frame
        x_pos[:, 0] += width/2.0
        x_pos[:, 1] += height/2.0
        x_pos[0:, 1] = height - x_pos[0:, 1]

        # Clear screen
        screen.fill(black)
        # Draw lines
        pygame.draw.line(screen, red, x_pos[0], x_pos[1])
        pygame.draw.line(screen, green, x_pos[1], x_pos[2])
        pygame.draw.line(screen, blue, x_pos[2], x_pos[3])
        # Show drawn objects
        pygame.display.flip()

        # FPS control
        end_time = pygame.time.get_ticks()
        pygame.time.wait(20 - (end_time - start_time))  # 50fps


if __name__ == "__main__":
    main()
