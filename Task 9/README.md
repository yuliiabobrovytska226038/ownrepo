Opentron OT-2 is simulated in pybullet v3.2 enviroment using python v3.12
At the start of the simulation manual command is disabled and the robot automaticaly scans its working envelope, moving to each of the 8 corners and printing
the coordinates of each point to the terminal.

corner 1 (-0.26, -0.18, 0.05)
corner 2 (-0.26, -0.18, 0.17)
corner 3 (-0.26, 0.26, 0.05)
corner 4 (-0.26, 0.26, 0.17)
corner 5 (0.13, -0.18, 0.05)
corner 6 (0.13, -0.18, 0.17)
corner 7 (0.13, 0.26, 0.05)
corner 8 (0.13, 0.26, 0.17)

After the robot finish scaning, Manual command mode is enabled by the 3 sliders controlling each axis manually, the value of each axis appears on top of the slider and
a button named "print endeffector position" print the current position to the terminal when pressed.