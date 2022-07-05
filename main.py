import numpy as np


class Robot():
    
    def __init__(self, origin):
        self.x_0 = origin
        self.link_length = np.array([80, 80, 80])
        self.w = np.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        self.w_inv = np.linalg.inv(self.w)

    def forward_kinematics(self, thetas):
        x_1 = self.x_0[0] + self.link_length[0]*np.cos(thetas[0,0])
        y_1 = self.x_0[1] + self.link_length[0]*np.sin(thetas[0,0])
        x_2 = x_1 + self.link_length[1]*np.cos(thetas[0,0] + thetas[1,0])
        y_2 = y_1 + self.link_length[1]*np.sin(thetas[0,0] + thetas[1,0])
        x_3 = x_2 + self.link_length[2]*np.cos(thetas[0,0] + thetas[1,0] + thetas[2,0])
        y_3 = y_2 + self.link_length[2]*np.sin(thetas[0,0] + thetas[1,0] + thetas[2,0])

        return np.array([self.x_0, [x_1, y_1], [x_2, y_2], [x_3, y_3]])
    
    def jacobian(self, thetas):
        jx_3 = -self.link_length[2]*np.sin(thetas[0,0]+thetas[1,0]+thetas[2,0])
        jx_2 = jx_3-self.link_length[1]*np.sin(thetas[0,0] + thetas[1,0])
        jx_1 = jx_2-self.link_length[0]*np.sin(thetas[0,0])
        jy_3 = self.link_length[2]*np.cos(thetas[0,0]+thetas[1,0]+thetas[2,0])
        jy_2 = jy_3+self.link_length[1]*np.cos(thetas[0,0] + thetas[1,0])
        jy_1 = jy_2+self.link_length[0]*np.cos(thetas[0,0])
        
        return np.matrix([[jx_1,jx_2,jx_3], 
                        [jy_1,jy_2,jy_3]])
    
    def inverse_kinematics(self, x_d, q):

        if np.linalg.norm(x_d) > np.sum(self.link_length):
            print("Position out of limits")
            return q

        x_current = np.transpose(np.matrix(self.forward_kinematics(q)[3]))
        x_inc = x_d - x_current 

        counter = 0
        while np.linalg.norm(x_inc) > 0.001:

            j = self.jacobian(q)
            j_t = np.transpose(j)
            pseudo_j = self.w_inv * j_t * np.linalg.inv(j * self.w_inv * j_t)

            q_inc = pseudo_j * x_inc
            q = q + q_inc
            x_current = np.transpose(np.matrix(self.forward_kinematics(q)[3]))
            x_inc =  x_d - x_current

            if counter > 100:
                break

            counter += 1
        
        return q
    
    def inverse_kinematics_null_space(self, x_d, q, q_star):

        if np.linalg.norm(x_d) > np.sum(self.link_length):
            print("Position out of limits")
            return q

        x_current = np.transpose(np.matrix(self.forward_kinematics(q)[3]))
        x_inc = x_d - x_current

        alpha = 0.5

        counter = 0
        while np.linalg.norm(x_inc) > 0.001:

            j = self.jacobian(q)
            j_t = np.transpose(j)
            pseudo_j = self.w_inv * j_t * np.linalg.inv(j * self.w_inv * j_t)
            Nw = np.identity(3) - pseudo_j * j

            q_inc = pseudo_j * x_inc - alpha * Nw * self.w_inv * (q - q_star)
            q = q + q_inc
            x_current = np.transpose(np.matrix(self.forward_kinematics(q)[3]))
            x_inc =  x_d - x_current

            if counter > 100:
                break

            counter += 1
        
        return q

    def inverse_kinematics_transpose(self, x_d, q):

        if np.linalg.norm(x_d) > np.sum(self.link_length):
            print("Position out of limits")
            return q

        x_current = np.transpose(np.matrix(self.forward_kinematics(q)[3]))
        x_inc = x_d - x_current 

        counter = 0
        while np.linalg.norm(x_inc) > 0.001:

            j = self.jacobian(q)
            j_t = np.transpose(j)

            q_inc = j_t * x_inc * 0.00001
            q = q + q_inc
            x_current = np.transpose(np.matrix(self.forward_kinematics(q)[3]))
            x_inc =  x_d - x_current

            if counter > 100:
                break

            counter += 1
        
        return q        
        
        

def main():
    import sys, pygame
    pygame.init()
    size = width, height = 640, 480
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    screen = pygame.display.set_mode(size)

    # Robot stuff
    origin = np.array([0.0, 0.0])
    robot = Robot(origin)
    q_pos = np.matrix([[0.1], [0.3], [0.2]])
    q_star = np.matrix([[np.pi/2.0], [0], [np.pi/2.0]])
    
    while True:
        start_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        
        # Robot stuff
        x_mouse, y_mouse = pygame.mouse.get_pos()
        x_mouse = x_mouse - width/2.0
        y_mouse = height/2.0 - y_mouse

        # print(x_mouse, y_mouse)

        q_pos = robot.inverse_kinematics_null_space(np.matrix([[x_mouse],[y_mouse]]), q_pos, q_star)
        x_pos = robot.forward_kinematics(q_pos)

        print(q_pos)

        #Coordinates adjustment
        x_pos[:,0] += width/2.0
        x_pos[:,1] += height/2.0
        x_pos[0:,1] = height - x_pos[0:,1]
        #

        screen.fill(black)
        pygame.draw.line(screen, red, x_pos[0], x_pos[1])
        pygame.draw.line(screen, green, x_pos[1], x_pos[2])
        pygame.draw.line(screen, blue, x_pos[2], x_pos[3])
        pygame.display.flip()

        end_time = pygame.time.get_ticks()
        pygame.time.wait(20 - (end_time - start_time) ) #50fps


if __name__ == "__main__":
    main()