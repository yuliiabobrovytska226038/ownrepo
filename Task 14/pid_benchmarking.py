import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pybullet as p
import pybullet_data
import time
import threading
import queue
import math
import numpy as np

########################################
##########/Setting Enviroment\##########
########################################

# Create a tkinter window
root = tk.Tk()
root.title("PID Control PyBullet")

# Connect to the physics client (use p.DIRECT for non-graphical version)
physicsClient = p.connect(p.GUI)
num_agents = 1
phsx_time_step = 1./240.

# Add search path for URDFs provided by pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set the gravity in the simulation
p.setGravity(0, 0, -9.81)

# Set the camera parameters
cameraDistance = 1.1 * (math.ceil((num_agents)**0.3))  # Distance from the target (zoom)
cameraYaw = 0  # Rotation around the vertical axis in degrees
cameraPitch = -35  # Rotation around the horizontal axis in degrees
cameraTargetPosition = [0, -(math.ceil(num_agents**0.5)/2)+0.5, 0.1]  # XYZ coordinates of the target position

# Reset the camera with the specified parameters
p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

# Load a plane URDF to serve as the ground
planeId = p.loadURDF("plane.urdf")

# Define the starting position and orientation for the OT-2 robot
startPos = [0, 0, 0]
end_effector_pos = (0,0,0)
pipette_offset = (0.06626509130001068, 0.2950528562068939, 0.5975000381469727)
startOrientation = p.getQuaternionFromEuler([0, 0, -1.57])

# Load the OT-2 robot URDF
OT2Id = p.loadURDF("Y2B-2023-OT2_Twin/ot_2_simulation_v6.urdf", startPos, startOrientation, useFixedBase=True)

# Create a matplotlib figure with 3 subplots arranged vertically
fig = Figure(figsize=(6, 6), dpi=100)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)


ax1.set_ylabel("X Position")
ax2.set_ylabel("Y Position")
ax3.set_ylabel("Z Position")
ax3.set_xlabel("Time (s)")

ax1.set_ylim(-0.3, 0.3)
ax2.set_ylim(-0.3, 0.3)
ax3.set_ylim(0, 0.2)

# Initialize the data arrays
time_data = []
x_data = []
y_data = []
z_data = []

x_Edata = []
y_Edata = []
z_Edata = []

# Create lines for each subplot
line1, = ax1.plot(time_data, x_data, 'r-', linewidth=0.5, linestyle='dashed')
line2, = ax2.plot(time_data, y_data, 'g-', linewidth=0.5, linestyle='dashed')
line3, = ax3.plot(time_data, z_data, 'b-', linewidth=0.5, linestyle='dashed')

line1E, = ax1.plot(time_data, x_Edata, 'r-')
line2E, = ax2.plot(time_data, y_Edata, 'g-')
line3E, = ax3.plot(time_data, z_Edata, 'b-')

# Create a canvas to display the figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# Create a new window for benchmarking graphs
benchmark_window = tk.Toplevel(root)
benchmark_window.title("Benchmarking")

# Create a matplotlib figure for benchmarking with 2 subplots
benchmark_fig = Figure(figsize=(6, 4), dpi=100)
ax_time = benchmark_fig.add_subplot(211)
ax_accuracy = benchmark_fig.add_subplot(212)

ax_time.set_ylabel("Execution Time (s)")
ax_accuracy.set_ylabel("Accuracy Error (m)")
ax_accuracy.set_xlabel("Time (s)")

# Initialize the data arrays for benchmarking
benchmark_time_data = []
execution_time_data = []
accuracy_data = []

# Create lines for benchmarking plots
line_time, = ax_time.plot(benchmark_time_data, execution_time_data, 'r-')
line_accuracy, = ax_accuracy.plot(benchmark_time_data, accuracy_data, 'b-')

# Create a canvas to display the benchmarking figure
benchmark_canvas = FigureCanvasTkAgg(benchmark_fig, master=benchmark_window)
benchmark_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# Queue to communicate between threads
queue_data = queue.Queue()
benchmark_queue_data = queue.Queue()

###############################
##########/Functions\##########
###############################

def get_tuble_subtraction(tup1, tup2):
    return tuple(map(lambda i, j: i - j, tup1, tup2))

def get_tuble_addition(tup1, tup2):
    return tuple(map(lambda i, j: i + j, tup1, tup2))

def get_abs_error(tub1, tub2, max_error):
    X,Y,Z = tub1; x,y,z = tub2
    return (abs(X-x)<=max_error) and (abs(Y-y)<=max_error) and (abs(Z-z)<=max_error)

# Function to get the end effector position
def get_end_effector_position(robot_id):
    # Assuming the end effector is the last link
    end_effector_state = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1)
    return get_tuble_subtraction(end_effector_state[4], pipette_offset)  # Position is the 5th element in the tuple

# Function to get the upper and lower limits of a joint
def get_joint_limits(robot_id, joint_index):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_lower_limit = joint_info[8]  # Lower limit is at index 8
    joint_upper_limit = joint_info[9]  # Upper limit is at index 9
    return joint_lower_limit, joint_upper_limit

# Function to calculate and get the PID values
def get_PID(constants, set_point, cur_point, integral, prev_error):
    Kp, Ki, Kd = constants

    error = set_point - cur_point # Calculate error
    integral += error * phsx_time_step # Calculate error integration
    derivative = (error - prev_error) / phsx_time_step # Calculate error derivative

    P = Kp * error
    I = Ki * integral
    D = Kd * derivative
    PID = P + I + D
    return PID, integral, error

# Function to calculate error
def calculate_error(desired_pos, actual_pos):
    return np.linalg.norm(np.array(desired_pos) - np.array(actual_pos))

###############################
##########/Robot SIM\##########
###############################

# Max Robot velocity and power
MVel_slider = p.addUserDebugParameter(" Actuator max velocity m/s", 0, 5, 1)
MPwr_slider = p.addUserDebugParameter(" Actuator max Power Watt", 0, 120, 25)

# Assuming the joint indices are known:
x_joint_index = 1  # index for gantry_x1
y_joint_index = 0  # index for gantry_y1
z_joint_index = 2  # index for gantry_z1

# Set the actual range for each slider (replace with actual ranges)
x_min, x_max = get_joint_limits(OT2Id, x_joint_index)  # Example range, replace with actual range for gantry_x1
y_min, y_max = get_joint_limits(OT2Id, y_joint_index)  # Example range, replace with actual range for gantry_y1
z_min, z_max = get_joint_limits(OT2Id, z_joint_index)  # Example range, replace with actual range for gantry_z1

# Create sliders for the gantry actuators
x_slider = p.addUserDebugParameter(" Gantry X", x_min, x_max, 0)
y_slider = p.addUserDebugParameter(" Gantry Y", y_min, y_max, 0)
z_slider = p.addUserDebugParameter(" Gantry Z", z_min, z_max, 0)

# Create sliders for the PID constants
P_slider = p.addUserDebugParameter(" P", 0, 100, 20)
I_slider = p.addUserDebugParameter(" I", 0, 10, 0.05)
D_slider = p.addUserDebugParameter(" D", 0, 10, 2.5)

# Create a button
button = p.addUserDebugParameter(" Print EndEffector error", 1, 0, 0)
prev_button_state = 0

# PID Controller state
integral_X = 0.0
integral_Y = 0.0
integral_Z = 0.0

previous_error_X = 0.0
previous_error_Y = 0.0
previous_error_Z = 0.0

######################################
##########/Physics SIM Step\##########
######################################

# Simulate
def update():

    global integral_X, integral_Y, integral_Z
    global previous_error_X, previous_error_Y, previous_error_Z, prev_button_state

    while True:
        loop_start_time = time.time()

        current_time = loop_start_time - start_time

        # End Effector actual position at this instant
        x_ee, y_ee, z_ee = get_end_effector_position(OT2Id)

        # Read slider values
        Max_Velocity = p.readUserDebugParameter(MVel_slider)
        Max_Power = p.readUserDebugParameter(MPwr_slider)

        x_pos = p.readUserDebugParameter(x_slider)
        y_pos = p.readUserDebugParameter(y_slider)
        z_pos = p.readUserDebugParameter(z_slider)

        K_P = p.readUserDebugParameter(P_slider)
        K_I = p.readUserDebugParameter(I_slider)
        K_D = p.readUserDebugParameter(D_slider)

        x_vel, integral_X, previous_error_X = get_PID((K_P, K_I, K_D), x_pos, x_ee, integral_X, previous_error_X)
        y_vel, integral_Y, previous_error_Y = get_PID((K_P, K_I, K_D), y_pos, y_ee, integral_Y, previous_error_Y)
        z_vel, integral_Z, previous_error_Z = get_PID((K_P, K_I, K_D), z_pos, z_ee, integral_Z, previous_error_Z)


        # Clip velocity to max velocity
        x_vel = max(min(x_vel, Max_Velocity), -Max_Velocity)
        y_vel = max(min(y_vel, Max_Velocity), -Max_Velocity)
        z_vel = max(min(z_vel, Max_Velocity), -Max_Velocity)

        # Set velocity to the actuator controlling the position with PID control 
        p.setJointMotorControl2(bodyIndex=OT2Id, jointIndex=x_joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=x_vel, force=(Max_Power/Max_Velocity))
        p.setJointMotorControl2(bodyIndex=OT2Id, jointIndex=y_joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=y_vel, force=(Max_Power/Max_Velocity))
        p.setJointMotorControl2(bodyIndex=OT2Id, jointIndex=z_joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=z_vel, force=(10*Max_Power/Max_Velocity))

        # Check button state
        button_state = p.readUserDebugParameter(button)
        if button_state != prev_button_state and (button_state - prev_button_state) == 1:
            print(f":Error in meters {previous_error_X, previous_error_Y, previous_error_Z}") # Get the end effector position
        prev_button_state = button_state

        # Calculate accuracy error
        accuracy_error = calculate_error((x_pos, y_pos, z_pos), (x_ee, y_ee, z_ee))

        # Put the data into the queue
        queue_data.put((current_time, x_pos, y_pos, z_pos, x_ee, y_ee, z_ee))
        benchmark_queue_data.put((current_time, time.time() - loop_start_time, accuracy_error))

        # Step the simulation
        p.stepSimulation()
        time.sleep(phsx_time_step)

# Start time for tracking elapsed time
start_time = time.time()

# Function to draw the canvas from the queue
def draw_canvas():
    while True:
        try:
            # Get data from the queue
            data = queue_data.get(timeout=0.015)
            td, xd, yd, zd, xe, ye, ze = data

            # Append the new data point
            time_data.append(td)
            x_data.append(xd)
            y_data.append(yd)
            z_data.append(zd)

            x_Edata.append(xe)
            y_Edata.append(ye)
            z_Edata.append(ze)

            # Limit the number of data points to 5000
            if len(time_data) > 5000:
                time_data.pop(0)
                x_data.pop(0)
                y_data.pop(0)
                z_data.pop(0)
                x_Edata.pop(0)
                y_Edata.pop(0)
                z_Edata.pop(0)

            # Update the plot
            line1.set_data(time_data, x_data)
            line2.set_data(time_data, y_data)
            line3.set_data(time_data, z_data)

            line1E.set_data(time_data, x_Edata)
            line2E.set_data(time_data, y_Edata)
            line3E.set_data(time_data, z_Edata)

            # Draw the canvas and set the axis limit
            if td % 1 <= 0.01:
                ax1.set_xlim(min(time_data), max(time_data))
                ax2.set_xlim(min(time_data), max(time_data))
                ax3.set_xlim(min(time_data), max(time_data))
                canvas.draw()
        except queue.Empty:
            continue

# Function to draw the benchmarking canvas from the queue
def draw_benchmark_canvas():
    while True:
        try:
            # Get data from the queue
            data = benchmark_queue_data.get(timeout=0.015)
            td, exec_time, accuracy = data

            # Append the new data point
            benchmark_time_data.append(td)
            execution_time_data.append(exec_time)
            accuracy_data.append(accuracy)

            # Limit the number of data points to 5000
            if len(benchmark_time_data) > 5000:
                benchmark_time_data.pop(0)
                execution_time_data.pop(0)
                accuracy_data.pop(0)

            # Update the plot
            line_time.set_data(benchmark_time_data, execution_time_data)
            line_accuracy.set_data(benchmark_time_data, accuracy_data)

            # Draw the canvas and set the axis limit
            if td % 1 <= 0.01:
                ax_time.set_xlim(min(benchmark_time_data), max(benchmark_time_data))
                ax_accuracy.set_xlim(min(benchmark_time_data), max(benchmark_time_data))
                benchmark_canvas.draw()
        except queue.Empty:
            continue

# Start the threads
threading.Thread(target=update, daemon=True).start()
threading.Thread(target=draw_canvas, daemon=True).start()
threading.Thread(target=draw_benchmark_canvas, daemon=True).start()

# Start the tkinter main loop
root.mainloop()

# Disconnect from PyBullet when tkinter window is closed
p.disconnect()
