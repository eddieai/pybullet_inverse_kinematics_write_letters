import pybullet as p
import time
import math
import numpy as np
from datetime import datetime
from trajectory import *

clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
    p.connect(p.GUI)

p.loadURDF("plane.urdf", [0, 0, -0.3])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7):
    exit()

useNullSpace = 1
useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 1
# if (useSimulation and useRealTimeSimulation == 0):
#     p.stepSimulation()
p.setRealTimeSimulation(useRealTimeSimulation)
p.setGravity(0, 0, 0)
# trailDuration is duration (in seconds) after debug lines will be removed automatically
trailDuration = 9

ikSolver = 0
# end effector points down, not up (in case useOrientation==1)
orn = p.getQuaternionFromEuler([0, -math.pi, 0])
# lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, 0, 0, 0, 0, 0, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
    p.resetJointState(kukaId, i, rp[i])


def IK_darw(letter='A', traj_regression='GMR', traj_plan='vertical', traj_resample=0.7):

    # trajectory of letter
    if traj_regression == 'GMR':
        traj = GMR_traj(letter)
    elif traj_regression == 'DMP_GMR':
        traj = DMP_GMR_traj(letter)
    elif traj_regression == 'LWR':
        traj = LWR_traj(letter)
    else:
        raise('trajectory regression type error')

    # add some frames at the start position of trajetory
    traj = np.concatenate((np.tile(traj[0], (int(traj_resample*20000), 1)), np.array(traj)))

    # resample trajetory to make it slower
    traj_1 = - np.interp(np.arange(0, len(traj), traj_resample), np.arange(0, len(traj)), traj[:, 0])
    traj_2 = np.interp(np.arange(0, len(traj), traj_resample), np.arange(0, len(traj)), traj[:, 1])

    # rescale trajectory to adapt to robot working area
    traj_1 = np.interp(traj_1, (np.min(traj_1), np.max(traj_1)), (-0.15, 0.15))
    traj_2 = np.interp(traj_2, (np.min(traj_2), np.max(traj_2)), (0.3, 0.6))

    prevPose = [0, 0, 0]
    prevPose1 = [0, 0, 0]
    for n in range(len(traj_1)):
        # Convert 2D letter trajecotry to 3D trajecotry in vertical plan or horizonal plan
        if traj_plan == 'vertical':
            pos = [-0.4, traj_1[n], traj_2[n]]
        elif traj_plan == 'horizontal':
            pos = [traj_1[n]-0.3, traj_2[n], 0.4]
        else:
            raise('trajectory plan error')

        if (useNullSpace == 1):
            if (useOrientation == 1):
                jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                      jr, rp)
            else:
                jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      lowerLimits=ll,
                                                      upperLimits=ul,
                                                      jointRanges=jr,
                                                      restPoses=rp)
        else:
            if (useOrientation == 1):
                jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      orn,
                                                      jointDamping=jd,
                                                      solver=ikSolver,
                                                      maxNumIterations=100,
                                                      residualThreshold=.01)
            else:
                jointPoses = p.calculateInverseKinematics(kukaId,
                                                      kukaEndEffectorIndex,
                                                      pos,
                                                      solver=ikSolver)

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
        else:
            #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for i in range(numJoints):
                p.resetJointState(kukaId, i, jointPoses[i])

        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        if n > 20000:
            p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 2, trailDuration)
            p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 5, trailDuration)
        prevPose = pos
        prevPose1 = ls[4]


IK_darw(letter='H', traj_regression='LWR', traj_plan='vertical', traj_resample=0.7)
IK_darw(letter='E', traj_regression='LWR', traj_plan='vertical', traj_resample=0.7)
IK_darw(letter='L', traj_regression='GMR', traj_plan='vertical', traj_resample=0.7)
IK_darw(letter='L', traj_regression='GMR', traj_plan='vertical', traj_resample=0.7)
IK_darw(letter='O', traj_regression='DMP_GMR', traj_plan='vertical', traj_resample=0.7)

IK_darw(letter='W', traj_regression='LWR', traj_plan='horizontal', traj_resample=0.7)
IK_darw(letter='O', traj_regression='LWR', traj_plan='horizontal', traj_resample=0.7)
IK_darw(letter='R', traj_regression='GMR', traj_plan='horizontal', traj_resample=0.7)
IK_darw(letter='L', traj_regression='GMR', traj_plan='horizontal', traj_resample=0.7)
IK_darw(letter='D', traj_regression='DMP_GMR', traj_plan='horizontal', traj_resample=0.7)

p.disconnect()