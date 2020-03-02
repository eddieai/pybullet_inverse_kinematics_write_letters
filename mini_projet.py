import pybullet as p
import time
import math
import numpy as np
from datetime import datetime
from trajectory import *


# pybullet 3D simulator parameters initialization
clid = p.connect(p.SHARED_MEMORY)
if (clid < 0): p.connect(p.GUI)
p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7): exit()
useSimulation = 1           # use dynamic motor control (can interact with robot in simulator)
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
p.setGravity(0, 0, 0)
trailDuration = 15          # trailDuration is duration (in seconds) after debug lines will be removed automatically

# Inverse Kinematics control's parameters
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]  # lower limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]        # upper limits for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]                    # joint ranges for null space
rp = [0, 0, 0, 0, 0, 0, 0]                          # restposes for null space
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]            # joint damping coefficients
ikSolver = 0

# Initialize robot joint states
for i in range(numJoints):
    p.resetJointState(kukaId, i, rp[i])


# Define IK_write function: Inverse Kinematics letter writing
def IK_write(letter='A', regression='GMR', n_states=20, plan='vertical', orientation='front', useNullSpace=True, resample=0.7):
    '''
    :param letter: str, Latin letter for robot to write
    :param regression: str, Letter trajectory regression method
    :param n_states: int, Number of states used in trajectory regression method
    :param plan: str, Robot writing letter in vertical or horizontal plan
    :param orientation: str, Robot's end effector point orientation
    :param useNullSpace: bool, Whether use NullSpace in inverse kinematics control or not
    :param resample: float, Resample rate of letter trajectory from regression method, use it for value between 0 and 1 to slow down the robot writing
    '''

    # Get trajectory letter using different regression types
    if regression == 'GMR':
        traj = GMR_traj(letter, n_states)       # Use GMR_traj fuction in trajectory.py to generate trajectory
    elif regression == 'DMP_GMR':
        traj = DMP_GMR_traj(letter, n_states)   # Use DMP_GMR_traj fuction in trajectory.py to generate trajectory
    elif regression == 'LWR':
        traj = LWR_traj(letter, n_states)       # Use LWR_traj fuction in trajectory.py to generate trajectory
    else:
        raise('Letter trajectory regression type parameter error')

    # Add some frames at the start position of trajetory, in order for robot to pause a bit before writing new letter
    traj = np.concatenate((np.tile(traj[0], (int(resample*2000), 1)), np.array(traj)))

    # Resample trajetory to make robot writing slower
    traj_1 = - np.interp(np.arange(0, len(traj), resample), np.arange(0, len(traj)), traj[:, 0])
    traj_2 = np.interp(np.arange(0, len(traj), resample), np.arange(0, len(traj)), traj[:, 1])

    # Rescale trajectory to adapt to robot working area
    traj_1 = np.interp(traj_1, (np.min(traj_1), np.max(traj_1)), (-0.15, 0.15))
    traj_2 = np.interp(traj_2, (np.min(traj_2), np.max(traj_2)), (0.3, 0.6))

    if orientation == 'front':
        orn = p.getQuaternionFromEuler([0, -0.5 * math.pi, 0])  # Robot's end effector point orientation (point down)
    elif orientation == 'down':
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # Robot's end effector point orientation (point down)
    else:
        raise ('Robot end effector orientation parameter error')

    prevPose = [0, 0, 0]
    prevPose1 = [0, 0, 0]
    # Begin robot writing letter
    for n in range(len(traj_1)):
        # Convert 2D letter trajecotry to 3D trajecotry in vertical plan or horizonal plan
        if plan == 'vertical':
            pos = [-0.6, traj_1[n], traj_2[n]]
        elif plan == 'horizontal':
            pos = [traj_1[n]-0.3, traj_2[n], 0.4]
        else:
            raise('Robot trajectory plan parameter error')

        if useNullSpace == True:
            # Calculate robot's joint poses using null-space inverse kinematics control
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      orn,
                                                      ll,
                                                      ul,
                                                      jr,
                                                      rp)
        else:
            # Calculate robot's joint poses using inverse kinematics without null-space control (use joint damping instead)
            jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      orn,
                                                      jointDamping=jd,
                                                      solver=ikSolver,
                                                      maxNumIterations=100,
                                                      residualThreshold=.01)

        # Set dynamic motion control for each joint
        if (useSimulation):
            for i in range(numJoints):
                p.setJointMotorControl2(bodyIndex=kukaId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
        # No use dynamic motion control
        else:
            #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for i in range(numJoints):
                p.resetJointState(kukaId, i, jointPoses[i])

        # Get link state of robot arm
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        # Draw target letter trajectory and robot trajectory inside simulator
        if n > 2000:
            p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 2, trailDuration)   # Draw red line of robot trajectory
            p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 5, trailDuration)    # Draw black line of target letter trajectory
        prevPose = pos
        prevPose1 = ls[4]


# Use IK_write to write each letter in 'HELLO' in vertical plan
IK_write(letter='H', regression='LWR', n_states=20, plan='vertical', orientation='front', useNullSpace=True, resample=0.7)
IK_write(letter='E', regression='LWR', n_states=20, plan='vertical', orientation='front', useNullSpace=True, resample=0.7)
IK_write(letter='L', regression='GMR', n_states=20, plan='vertical', orientation='front', useNullSpace=True, resample=0.7)
IK_write(letter='L', regression='GMR', n_states=20, plan='vertical', orientation='front', useNullSpace=True, resample=0.7)
IK_write(letter='O', regression='DMP_GMR', n_states=20, plan='vertical', orientation='front', useNullSpace=True, resample=0.7)

# Use IK_write to write each letter in 'WORLD' in horizontal plan
IK_write(letter='W', regression='LWR', n_states=20, plan='horizontal', orientation='down', useNullSpace=True, resample=0.7)
IK_write(letter='O', regression='LWR', n_states=20, plan='horizontal', orientation='down', useNullSpace=True, resample=0.7)
IK_write(letter='R', regression='GMR', n_states=20, plan='horizontal', orientation='down', useNullSpace=True, resample=0.7)
IK_write(letter='L', regression='GMR', n_states=20, plan='horizontal', orientation='down', useNullSpace=True, resample=0.7)
IK_write(letter='D', regression='DMP_GMR', n_states=20, plan='horizontal', orientation='down', useNullSpace=True, resample=0.7)

p.disconnect()