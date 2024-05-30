import pybullet as p
import time
import pybullet_data
import math



########################################
##########/Setting Enviroment\##########
########################################

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
Velocity = 0.1 # motor velocity in m/s
manual_Control = False # activate slider manual control or not

# Load the OT-2 robot URDF
OT2Id = p.loadURDF("Y2B-2023-OT2_Twin/ot_2_simulation_v6.urdf", startPos, startOrientation, useFixedBase=True)




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

def execute_trajectory(robot_id, desired_pos, Vel):
    while not get_abs_error(desired_pos, get_end_effector_position(robot_id), 5e-4):
        p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=x_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=desired_pos[0], maxVelocity=Vel)
        p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=y_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=desired_pos[1], maxVelocity=Vel)
        p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=z_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=desired_pos[2], maxVelocity=Vel)

        # Step the simulation
        p.stepSimulation()
        time.sleep(phsx_time_step)
    print("corner hit", desired_pos)
    return





###############################
##########/Robot SIM\##########
###############################

# Assuming the joint indices are known:
x_joint_index = 1  # index for gantry_x1
y_joint_index = 0  # index for gantry_y1
z_joint_index = 2  # index for gantry_z1

# Set the actual range for each slider (replace with actual ranges)
x_min, x_max = get_joint_limits(OT2Id, x_joint_index)  # Example range, replace with actual range for gantry_x1
y_min, y_max = get_joint_limits(OT2Id, y_joint_index)  # Example range, replace with actual range for gantry_y1
z_min, z_max = get_joint_limits(OT2Id, z_joint_index)  # Example range, replace with actual range for gantry_z1

# Create sliders for the gantry actuators
x_slider = p.addUserDebugParameter("Gantry X", x_min, x_max, 0)
y_slider = p.addUserDebugParameter("Gantry Y", y_min, y_max, 0)
z_slider = p.addUserDebugParameter("Gantry Z", z_min, z_max, 0)

# Define the 8 corner positions of the cube
corner_positions = [(x, y, z) for x in [x_min, x_max] for y in [y_min, y_max] for z in [z_min, z_max]]

# Create a button
button = p.addUserDebugParameter("Print EndEffector Pos", 1, 0, 0)
prev_button_state = 0



######################################
##########/Physics SIM Step\##########
######################################

# Simulate
while 1:
    # Read slider values
    x_pos = p.readUserDebugParameter(x_slider)
    y_pos = p.readUserDebugParameter(y_slider)
    z_pos = p.readUserDebugParameter(z_slider)
    
    if manual_Control:
        # Set the joint positions according to the slider values
        p.setJointMotorControl2(bodyUniqueId=OT2Id, jointIndex=x_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=x_pos, maxVelocity=Velocity)
        p.setJointMotorControl2(bodyUniqueId=OT2Id, jointIndex=y_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=y_pos, maxVelocity=Velocity)
        p.setJointMotorControl2(bodyUniqueId=OT2Id, jointIndex=z_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=z_pos, maxVelocity=Velocity)

        # Check button state
        button_state = p.readUserDebugParameter(button)
        if button_state != prev_button_state and (button_state - prev_button_state) == 1:
            print(f"End Effector Position: {get_end_effector_position(OT2Id)}") # Get the end effector position
        prev_button_state = button_state

    else:
        # remove old text Add a label to the GUI
        for corner_position in corner_positions:
            # Plan trajectory to the corner position (e.g., using IK or Cartesian planning)
            execute_trajectory(OT2Id, corner_position, 0.2) # Execute the trajectory
            
        manual_Control = True


    # Step the simulation
    p.stepSimulation()
    time.sleep(phsx_time_step)

# Disconnect from the physics client
p.disconnect()
