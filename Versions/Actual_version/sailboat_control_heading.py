#!/usr/bin/env python
# license removed for brevity

import rospy
import math
import tf
import communicate as cm
import text_file as db
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist, Point, Quaternion
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler

initial_pose = Odometry()
target_pose = Odometry()
rate_value = 2   # Period of saving data
control_rate = 1 # Period of the controller (1/control_time)
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
save_data = False
counter = 0
time_counter = 0
base = ''
db_name = 'Test_1'
constant = [2,3]
state_msg = ''
yaw_angles = [0,135,179,0] #Train yaw angles
yaw_counter = 0

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
    global counter
    global time_counter
    global base 
    global constant

    rospy.init_node('usv_simple_ctrl', anonymous=True)
    rate = rospy.Rate(rate_value) # 0.5Hz
    # publishes to thruster and rudder topics
    pub_sail = rospy.Publisher('/sail/angleLimits', Float64, queue_size=10)
    #pub_sail_2 = rospy.Publisher('sail_2/angleLimits', Float64, queue_size=10)
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
	    time_counter += (1.0/rate_value)
	    if save_data and constant[0]!=constant[1]:
                base = db.text_files(new_file = True, file_name = db_name, structure = ['time','speed','wind','x','y'])  
		constant[0] = constant[1]
		time_counter = 0
	    final = rudder_ctrl_msg()
	    if not save_data or counter >= (rate_value // control_rate):
                pub_rudder.publish(final[0])
	        pub_sail.publish(final[1])
		counter = 0
                #pub_sail_2.publish(final[2])
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

def reset_environment(yaw):
    q = quaternion_from_euler(0, 0, yaw)
    state_msg.model_name = 'sailboat'
    state_msg.pose.position.x = 240
    state_msg.pose.position.y = 100
    state_msg.pose.position.z = 0
    state_msg.pose.orientation.x = q[0]
    state_msg.pose.orientation.y = q[1]
    state_msg.pose.orientation.z = q[2]
    state_msg.pose.orientation.w = q[3]
    resp = set_state(state_msg)

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
    global base
    global counter
    global time_counter
    global constant
    global db_name
    global yaw_counter
	
    port_name='interface_2'
    direction= '/home/nelson/Documentos/Ubuntu_master/SNN_Codes/Spiking_codes'
    timeout=15
    rudder_angle=0
    sail_angle = 1
    sail_angle_2 = 1
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
    counter +=1	
    if save_data:
	speed = math.sqrt(x2**2+y2**2)
	base.append_data([time_counter,speed,math.degrees(global_dir),x1,y1])
    # Send all the position sensors information to controller in python 3
    if not save_data or counter >= (rate_value // control_rate):
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
                    result_py3=recibe['A4']
                    rudder_angle=recibe['A1']
                    sail_angle=recibe['A2']
                    sail_angle_2=recibe['A3']
		    if result_py3 == 2:
		        #reset_world()
			reset_environment(math.radians(yaw_angles[yaw_counter]))
	                result_py3=0
		    elif result_py3 == 1000:
			yaw_counter = 0
			reset_environment(math.radians(yaw_angles[yaw_counter]))
			result_py3=0
		    elif result_py3 > 100:
			yaw_counter += 1
			reset_environment(math.radians(yaw_angles[yaw_counter]))
			result_py3=0
		    elif result_py3 > 2 and save_data:
			db_name=db_name[0:len(db_name)-1]+str(int(result_py3-2))
			constant[1] = result_py3
			
			
        except:
            rospy.loginfo("Error abriendo el puerto")


        #############################################   

        # Actualization of result
        result.data = int(result_py3)
        #############################################  
        #Timon y vela
        return math.radians(rudder_angle),math.radians(sail_angle),math.radians(sail_angle_2)
    
    else:
	return 0,0,0

def rudder_ctrl_msg():
    msg = JointState()
    msg.header = Header()
    msg.name = ['rudder_joint', 'sail_joint', 'sail_joint_2']
    res = controller()
    msg.position = [res[0], res[1], res[2]]
    msg.velocity = []
    msg.effort = []
    return msg,res[1],res[2]

if __name__ == '__main__':

    global reset_world
    global control_rate
    global rate_value

    #rospy.wait_for_service('/gazebo/reset_world')
    #reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

    state_msg = ModelState()
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    reset_environment(math.radians(yaw_angles[yaw_counter]))

    if not save_data:
	rate_value = control_rate
    try:
        talker_ctrl()
    except rospy.ROSInterruptException:
        pass
