import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pybullet as p
import pybullet_data
import time
import threading
import queue
import math

########################################
##########/Setting Environment\##########
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

# Queue to communicate between threads
queue_data = queue.Queue()

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

# Function to get the joint limits
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

###############################
##########/Robot SIM\##########
###############################

# Assuming the joint indices are known:
x_joint_index = 1  # index for gantry_x1
y_joint_index = 0  # index for gantry_y1
z_joint_index = 2  # index for gantry_z1

# Set the actual range for each slider (replace with actual ranges)
x_min, x_max = get_joint_limits(OT2Id, x_joint_index)
y_min, y_max = get_joint_limits(OT2Id, y_joint_index)
z_min, z_max = get_joint_limits(OT2Id, z_joint_index)

# Define desired coordinates to move the robot
desired_x = 0.2
desired_y = 0.
desired_z = 0.15

# Create sliders for the gantry actuators
x_slider = p.addUserDebugParameter(" Gantry X", x_min, x_max, 0)
y_slider = p.addUserDebugParameter(" Gantry Y", y_min, y_max, 0)
z_slider = p.addUserDebugParameter(" Gantry Z", z_min, z_max, 0)

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

# Actuators specs
Max_Velocity = 1
Max_Power = 25

######################################
##########/Physics SIM Step\##########
######################################

# Simulate
def update():

    global integral_X, integral_Y, integral_Z
    global previous_error_X, previous_error_Y, previous_error_Z, prev_button_state

    while True:
        current_time = time.time() - start_time

        # End Effector actual position at this instant
        x_ee, y_ee, z_ee = get_end_effector_position(OT2Id)

        # Set desired positions directly
        x_pos = desired_x
        y_pos = desired_y
        z_pos = desired_z

        x_vel, integral_X, previous_error_X = get_PID((20, 0.05, 2.5), x_pos, x_ee, integral_X, previous_error_X)
        y_vel, integral_Y, previous_error_Y = get_PID((20, 0.05, 2.5), y_pos, y_ee, integral_Y, previous_error_Y)
        z_vel, integral_Z, previous_error_Z = get_PID((20, 0, 0), z_pos, z_ee, integral_Z, previous_error_Z)

        # Clip velocity to max velocity
        x_vel = max(min(x_vel, Max_Velocity), -Max_Velocity)
        y_vel = max(min(y_vel, Max_Velocity), -Max_Velocity)
        z_vel = max(min(z_vel, Max_Velocity), -Max_Velocity)

        # Set velocity to the actuators controlling the position with PID control
        p.setJointMotorControl2(bodyIndex=OT2Id, jointIndex=x_joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=x_vel, force=(Max_Power/Max_Velocity))
        p.setJointMotorControl2(bodyIndex=OT2Id, jointIndex=y_joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=y_vel, force=(Max_Power/Max_Velocity))
        # Set velocity to the actuators controlling the position with PID control
        p.setJointMotorControl2(bodyIndex=OT2Id, jointIndex=z_joint_index, controlMode=p.VELOCITY_CONTROL, targetVelocity=z_vel, force=(10*Max_Power/Max_Velocity))

        # Check button state
        button_state = p.readUserDebugParameter(button)
        if button_state != prev_button_state and (button_state - prev_button_state) == 1:
            print(f"Error in meters: {previous_error_X}, {previous_error_Y}, {previous_error_Z}")  # Print the end effector position error
        prev_button_state = button_state

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

# Start the threads
threading.Thread(target=update, daemon=True).start()
threading.Thread(target=draw_canvas, daemon=True).start()

# Start the tkinter main loop
root.mainloop()

# Disconnect from PyBullet when tkinter window is closed
p.disconnect()


#detailes data for each plant in image1
detailed_plants_data = [
    {
        'primary_root': {
            'V1': (950, 2590),
            'V2': (1135, 659),
            'length': 2087.987
        },
        'secondary_roots': [
            {'V1': (970, 2094), 'V2': (983, 2089), 'length': 15.071},
            {'V1': (980, 1753), 'V2': (1043, 1670), 'length': 112.61},
            {'V1': (1040, 1858), 'V2': (1063, 1876), 'length': 31.627},
            {'V1': (1046, 1016), 'V2': (1143, 748), 'length': 323.794},
            {'V1': (1100, 1282), 'V2': (1106, 1280), 'length': 6.828},
            {'V1': (1129, 1189), 'V2': (1136, 1192), 'length': 8.243},
            {'V1': (1139, 660), 'V2': (1365, 965), 'length': 416.772},
            {'V1': (1165, 797), 'V2': (1321, 989), 'length': 289.664},
            {'V1': (1180, 1079), 'V2': (1251, 1511), 'length': 469.794},
            {'V1': (1182, 890), 'V2': (1331, 1270), 'length': 498.144}
        ]
    },
    {
        'primary_root': {'length': 1832.311, 'V1': (1657, 581, 0), 'V2': (1536, 2311, 0)},
        'secondary_roots': [
            {'length': 51.355, 'V1': (1496, 1758, 0), 'V2': (1537, 1733, 0)},
            {'length': 6.243, 'V1': (1523, 1893, 0), 'V2': (1528, 1890, 0)},
            {'length': 7.828, 'V1': (1583, 1280, 0), 'V2': (1590, 1278, 0)},
            {'length': 210.451, 'V1': (1613, 1064, 0), 'V2': (1643, 926, 0)},
            {'length': 8.657, 'V1': (1622, 1081, 0), 'V2': (1629, 1077, 0)},
            {'length': 210.995, 'V1': (1662, 581, 0), 'V2': (1732, 753, 0)},
            {'length': 165.723, 'V1': (1887, 992, 0), 'V2': (1803, 864, 0)},
            {'length': 140.953, 'V1': (1672, 800, 0), 'V2': (1791, 853, 0)},
            {'length': 132.338, 'V1': (1654, 660, 0), 'V2': (1732, 753, 0)},
            {'length': 118.196, 'V1': (1733, 762, 0), 'V2': (1791, 853, 0)},
            {'length': 36.816, 'V1': (1828, 888, 0), 'V2': (1803, 864, 0)},
            {'length': 32.828, 'V1': (1731, 794, 0), 'V2': (1733, 762, 0)},
            {'length': 10.243, 'V1': (1732, 753, 0), 'V2': (1733, 762, 0)},
            {'length': 9.848, 'V1': (1798, 858, 0), 'V2': (1803, 864, 0)},
            {'length': 9.071, 'V1': (1791, 853, 0), 'V2': (1798, 858, 0)},
            {'length': 8.952, 'V1': (1798, 858, 0), 'V2': (1803, 864, 0)},
            {'length': 7.405, 'V1': (1803, 864, 0), 'V2': (1803, 864, 0)}
        ]
    },
    {
        'primary_root': {'length': 2000.276, 'V1': (1956, 2587, 0), 'V2': (2157, 723, 0)},
        'secondary_roots': [
            {'length': 498.434, 'V1': (1945, 1888, 0), 'V2': (1978, 1459, 0)},
            {'length': 25.485, 'V1': (1960, 1617, 0), 'V2': (1983, 1623, 0)},
            {'length': 190.267, 'V1': (1961, 1420, 0), 'V2': (1972, 1259, 0)},
            {'length': 165.368, 'V1': (2031, 1132, 0), 'V2': (1973, 1255, 0)},
            {'length': 117.385, 'V1': (1979, 1371, 0), 'V2': (1972, 1259, 0)},
            {'length': 59.113, 'V1': (2018, 1228, 0), 'V2': (1973, 1255, 0)},
            {'length': 4.414, 'V1': (1972, 1259, 0), 'V2': (1973, 1255, 0)},
            {'length': 32.799, 'V1': (1970, 1898, 0), 'V2': (1997, 1912, 0)},
            {'length': 114.225, 'V1': (2013, 957, 0), 'V2': (2105, 909, 0)},
            {'length': 574.546, 'V1': (2022, 1362, 0), 'V2': (2086, 1847, 0)},
            {'length': 36.87, 'V1': (2067, 1082, 0), 'V2': (2096, 1101, 0)},
            {'length': 504.926, 'V1': (2468, 1492, 0), 'V2': (2218, 1102, 0)},
            {'length': 198.149, 'V1': (2099, 955, 0), 'V2': (2214, 1100, 0)},
            {'length': 163.894, 'V1': (2085, 1020, 0), 'V2': (2214, 1100, 0)},
            {'length': 30.213, 'V1': (2238, 1121, 0), 'V2': (2218, 1102, 0)},
            {'length': 4.828, 'V1': (2214, 1100, 0), 'V2': (2218, 1102, 0)},
            {'length': 85.64, 'V1': (2104, 779, 0), 'V2': (2154, 717, 0)},
            {'length': 260.262, 'V1': (2159, 820, 0), 'V2': (2354, 962, 0)},
            {'length': 913.531, 'V1': (2574, 1368, 0), 'V2': (2228, 647, 0)},
            {'length': 29.385, 'V1': (2178, 663, 0), 'V2': (2202, 650, 0)}
        ]
    },
    {
        'primary_root': {'length': 1685.997, 'V1': (2634, 727, 0), 'V2': (2653, 2242, 0)},
        'secondary_roots': [
            {'length': 525.73, 'V1': (2428, 1554, 0), 'V2': (2615, 1119, 0)},
            {'length': 58.527, 'V1': (2567, 1909, 0), 'V2': (2592, 1862, 0)},
            {'length': 76.213, 'V1': (2595, 1910, 0), 'V2': (2586, 1980, 0)},
            {'length': 36.728, 'V1': (2573, 2026, 0), 'V2': (2582, 1993, 0)},
            {'length': 32.828, 'V1': (2583, 2024, 0), 'V2': (2582, 1993, 0)},
            {'length': 22.385, 'V1': (2600, 1964, 0), 'V2': (2586, 1980, 0)},
            {'length': 14.657, 'V1': (2582, 1993, 0), 'V2': (2586, 1980, 0)},
            {'length': 521.328, 'V1': (2574, 1354, 0), 'V2': (2605, 1755, 0)},
            {'length': 486.948, 'V1': (2590, 1473, 0), 'V2': (2733, 1887, 0)},
            {'length': 74.657, 'V1': (2602, 2091, 0), 'V2': (2606, 2018, 0)},
            {'length': 16.728, 'V1': (2606, 2082, 0), 'V2': (2615, 2069, 0)},
            {'length': 44.284, 'V1': (2609, 1978, 0), 'V2': (2629, 2014, 0)},
            {'length': 11, 'V1': (2618, 2080, 0), 'V2': (2618, 2091, 0)},
            {'length': 345.693, 'V1': (2640, 1188, 0), 'V2': (2644, 903, 0)},
            {'length': 760.477, 'V1': (3106, 1429, 0), 'V2': (2785, 805, 0)},
            {'length': 603.683, 'V1': (3023, 1306, 0), 'V2': (2785, 805, 0)},
            {'length': 142.882, 'V1': (2701, 714, 0), 'V2': (2781, 801, 0)},
            {'length': 123.066, 'V1': (2701, 714, 0), 'V2': (2781, 801, 0)},
            {'length': 57.113, 'V1': (2647, 735, 0), 'V2': (2695, 713, 0)},
            {'length': 21.971, 'V1': (2680, 703, 0), 'V2': (2695, 713, 0)},
            {'length': 6.414, 'V1': (2695, 713, 0), 'V2': (2701, 714, 0)},
            {'length': 6.243, 'V1': (2781, 801, 0), 'V2': (2785, 805, 0)}
        ]
    },
    {
        'primary_root': {'length': 1851.909, 'V1': (3198, 561, 0), 'V2': (3307, 2299, 0)},
        'secondary_roots': [
            {'length': 208.066, 'V1': (3112, 812, 0), 'V2': (3202, 650, 0)},
            {'length': 18.071, 'V1': (3183, 711, 0), 'V2': (3199, 706, 0)},
            {'length': 220.179, 'V1': (3183, 841, 0), 'V2': (3295, 1006, 0)},
            {'length': 178.622, 'V1': (3208, 728, 0), 'V2': (3316, 849, 0)}
        ]
    }
]





import math

def scale_coordinates(plants_data, max_value=3.0, min_distance=0.5):
    scaled_data = []
    previous_point = None
    
    for plant in plants_data:
        # Scale primary root V1
        v1 = plant['primary_root']['V1']
        if len(v1) == 3:
            x1, y1, _ = v1  # Unpack x, y, and ignore z
        else:
            x1, y1 = v1  # Unpack x, y
        scaled_v1 = (scale_coordinate(x1, max_value), scale_coordinate(y1, max_value), -1.0)
        scaled_data.append(scaled_v1)
        
        # Ensure significant distance from previous point
        if previous_point is not None and distance(scaled_v1, previous_point) < min_distance:
            scaled_v1 = adjust_point(scaled_v1, previous_point, min_distance)
        previous_point = scaled_v1
        
        # Scale primary root V2
        v2 = plant['primary_root']['V2']
        if len(v2) == 3:
            x2, y2, _ = v2  # Unpack x, y, and ignore z
        else:
            x2, y2 = v2  # Unpack x, y
        scaled_v2 = (scale_coordinate(x2, max_value), scale_coordinate(y2, max_value), -1.0)
        
        # Ensure significant distance from previous point
        if distance(scaled_v2, previous_point) < min_distance:
            scaled_v2 = adjust_point(scaled_v2, previous_point, min_distance)
        previous_point = scaled_v2
        
        scaled_data.append(scaled_v2)
        
        # Scale secondary roots
        for root in plant['secondary_roots']:
            # Scale V1
            v1 = root['V1']
            if len(v1) == 3:
                x1, y1, _ = v1  # Unpack x, y, and ignore z
            else:
                x1, y1 = v1  # Unpack x, y
            scaled_v1 = (scale_coordinate(x1, max_value), scale_coordinate(y1, max_value), -1.0)
            
            # Ensure significant distance from previous point
            if distance(scaled_v1, previous_point) < min_distance:
                scaled_v1 = adjust_point(scaled_v1, previous_point, min_distance)
            previous_point = scaled_v1
            
            scaled_data.append(scaled_v1)
            
            # Scale V2
            v2 = root['V2']
            if len(v2) == 3:
                x2, y2, _ = v2  # Unpack x, y, and ignore z
            else:
                x2, y2 = v2  # Unpack x, y
            scaled_v2 = (scale_coordinate(x2, max_value), scale_coordinate(y2, max_value), -1.0)
            
            # Ensure significant distance from previous point
            if distance(scaled_v2, previous_point) < min_distance:
                scaled_v2 = adjust_point(scaled_v2, previous_point, min_distance)
            previous_point = scaled_v2
            
            scaled_data.append(scaled_v2)
    
    return scaled_data

def scale_coordinate(coord, max_value):
    # Ensure the coordinate is in float format
    coord_float = float(coord)
    # Scale coordinate to range [-max_value, max_value]
    scaled_coord = (2.0 * coord_float / 65535.0 - 1.0) * max_value
    return scaled_coord

def distance(point1, point2):
    # Calculate Euclidean distance between two points in 3D space
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def adjust_point(point, reference_point, min_distance):
    # Adjust the point to ensure it has a minimum distance from the reference point
    direction_vector = (point[0] - reference_point[0], point[1] - reference_point[1], point[2] - reference_point[2])
    norm = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2)
    scaled_direction_vector = (direction_vector[0] / norm, direction_vector[1] / norm, direction_vector[2] / norm)
    adjusted_point = (reference_point[0] + scaled_direction_vector[0] * min_distance,
                      reference_point[1] + scaled_direction_vector[1] * min_distance,
                      reference_point[2] + scaled_direction_vector[2] * min_distance)
    return adjusted_point

scaled_coordinates_list = scale_coordinates(detailed_plants_data, max_value=3.0, min_distance=0.5)

# Print the scaled coordinates list for verification
print(scaled_coordinates_list)
print(scaled_coordinates_list[0])