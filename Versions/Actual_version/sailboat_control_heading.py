#!/usr/bin/env python
# license removed for brevity

import rospy
import math
import tf
import communicate as cm
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist, Point, Quaternion
from std_msgs.msg import Float64
from std_srvs.srv import Empty

initial_pose = Odometry()
target_pose = Odometry()
rate_value = 0.5
result = Float64()
result.data = 0
windDir= Float64()
windDir.data = 1.5 
currentHeading= Float64()
currentHeading.data = 0
current_heading = 0
heeling = 0
spHeading = 10 
isTacking = 0
reset_world = 0

def get_pose(initial_pose_tmp):
    global initial_pose 
    initial_pose = initial_pose_tmp

def get_target(target_pose_tmp):
    global target_pose 
    target_pose = target_pose_tmp
    
def angle_saturation(sensor):
    if sensor > 180:
        sensor = sensor - 360
    if sensor < -180:
        sensor = sensor + 360
    return sensor

def talker_ctrl():
    global rate_value
    global currentHeading
    global windDir 
    global isTacking
    global heeling
    global spHeading

    rospy.init_node('usv_simple_ctrl', anonymous=True)
    rate = rospy.Rate(rate_value) # 10Hz
    # publishes to thruster and rudder topics
    #pub_sail = rospy.Publisher('angleLimits', Float64, queue_size=10)
    pub_rudder = rospy.Publisher('joint_setpoint', JointState, queue_size=10)
    pub_result = rospy.Publisher('move_usv/result', Float64, queue_size=10)
    pub_heading = rospy.Publisher('currentHeading', Float64, queue_size=10)
    pub_windDir = rospy.Publisher('windDirection', Float64, queue_size=10)
    pub_heeling = rospy.Publisher('heeling', Float64, queue_size=10)
    pub_spHeading = rospy.Publisher('spHeading', Float64, queue_size=10)
    
    # subscribe to state and targer point topics
    rospy.Subscriber("state", Odometry, get_pose)  # get usv position (add 'gps' position latter)
    rospy.Subscriber("move_usv/goal", Odometry, get_target)  # get target position

    while not rospy.is_shutdown():
        try:
            pub_rudder.publish(rudder_ctrl_msg())
            pub_result.publish(result)
            pub_heading.publish(currentHeading)
            pub_windDir.publish(windDir)
            pub_heeling.publish(heeling)
            pub_spHeading.publish(spHeading)
            rate.sleep()
        except rospy.ROSInterruptException:
	    rospy.logerr("ROS Interrupt Exception! Just ignore the exception!")
        except rospy.ROSTimeMovedBackwardsException:
	    rospy.logerr("ROS Time Backwards! Just ignore the exception!")


def controller():
    # erro = sp - atual
    # ver qual gira no horario ou anti-horario
    # aciona o motor (por enquanto valor fixo)
    global initial_pose
    global target_pose
    global current_heading
    global currentHeading
    global spHeading
    global windDir
    global heeling
    global result
	
    port_name='interface_2'
    direction= '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes'
    timeout=15
    rudder_angle=0
    sail_angle = 1
    result_py3=0
    # Position(set up and GPS) and odometry sensors processing and aconditioning

    x1 = initial_pose.pose.pose.position.x
    y1 = initial_pose.pose.pose.position.y
    x2 = initial_pose.twist.twist.linear.x
    y2 = initial_pose.twist.twist.linear.y
    quaternion = (initial_pose.pose.pose.orientation.x, initial_pose.pose.pose.orientation.y, initial_pose.pose.pose.orientation.z,initial_pose.pose.pose.orientation.w) 
    euler = tf.transformations.euler_from_quaternion(quaternion)

    target_angle = math.degrees(euler[2])
    myradians = math.atan2(y2-y1,x2-x1)
    sp_angle = math.degrees(myradians)
    sp_angle = angle_saturation(sp_angle)
    spHeading = sp_angle
    target_angle = angle_saturation(target_angle)
    target_angle = -target_angle
    current_heading = math.radians(target_angle)
    currentHeading.data = current_heading
    ##############################################
    # Wind sensor processing and aconditionating
    x = rospy.get_param('/uwsim/wind/x')
    y = rospy.get_param('/uwsim/wind/y')
    global_dir = math.atan2(y,x)
    heeling = angle_saturation(math.degrees(global_dir)+180)
    wind_dir = global_dir + current_heading
    wind_dir = angle_saturation(math.degrees(wind_dir))
    windDir.data = math.radians(angle_saturation(math.degrees(wind_dir)+180))
    #############################################

    # Send all the position sensors information to controller in python 3
    try:
	cm.serial_initialization(direction,port_name,timeout)
	info=[]
	info.append(round(math.degrees(euler[0]),0)) 
	info.append(round(math.degrees(euler[1]),0)) 
	info.append(round(math.degrees(-current_heading),0)) 
	info.append(round(wind_dir,0)) 
        datos={'S1': x1, 'S2': y1, 'S3': x2, 'S4': y2, 'S5': info[0], 'S6': info[1], 'S7': info[2], 'S8': info[3]}
        if not cm.write_data(datos):
            rospy.loginfo("No se pudo escribir en el puerto")
        else:
            [band,recibe] = cm.read_data()
	    if band: 
                result_py3=recibe['A3']
                rudder_angle=recibe['A1']
                sail_angle=recibe['A2']
		if result_py3 == 2:
		    reset_world()
	            result_py3=0
    except:
        rospy.loginfo("Error abriendo el puerto")

    #############################################   

    # Actualization of result
    result.data = int(result_py3)
    #############################################  

    return math.radians(rudder_angle),math.radians(sail_angle)

def rudder_ctrl_msg():
    msg = JointState()
    msg.header = Header()
    msg.name = ['rudder_joint', 'sail_joint']
    res = controller()
    msg.position = [res[0], res[1]]
    msg.velocity = []
    msg.effort = []
    return msg

if __name__ == '__main__':

    global reset_world
    rospy.wait_for_service('/gazebo/reset_world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    try:
        talker_ctrl()
    except rospy.ROSInterruptException:
        pass
