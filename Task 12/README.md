PID control on prismatic actuators velocity over time to precisely control the position at given set point
PID = Kp*error + Ki*integ(error) + Kd*dedt

ziegler nichols method was used to tune the PID parameters in addition of manual tuning to get the best performance possible
the position error is well below 1mm at speed of 1m/s for x and y gantry, and max speed of 100mm/s for z gantry

The optimum values for PID parameters:
X_gantry => kp = 20, ki = 0.05, kd = 2.5, maxVel = 1m/s
X_gantry => kp = 20, ki = 0.05, kd = 2.5, maxVel = 1m/s
X_gantry => kp = 20, ki = 0, kd = 0, , maxVel = 0.1m/s