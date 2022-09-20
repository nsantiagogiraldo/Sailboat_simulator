import math

Kp = 1
Ki = 0
ianterior = 0
target_distance = 0
spHeading = 10
current_heading = 0
heeling = 0
windDir = 0
f_distance = 4
rate_value = 10


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z  # in radians


def angle_saturation(sensor):
    while sensor >180 or sensor < -180:
        if sensor > 180:
            sensor = sensor - 360
        if sensor < -180:
            sensor = sensor + 360
    return sensor


def P(erro):
    global Kp
    return Kp * erro


def Integral(erro):
    global Ki
    global ianterior
    global rate_value
    if (ianterior > 0 and erro < 0) or (ianterior < 0 and erro > 0):
        ianterior = ianterior + Ki * erro * 50 * (1. / rate_value)
    else:
        ianterior = ianterior + Ki * erro * (1. / rate_value)
    return ianterior


def rudder_ctrl(position, euler):  # position=[x1,y1,x2,y2]  orientation=[x,y,z,w]
    global target_distance
    global spHeading
    global current_heading
    myradians = math.atan2(position[3] - position[1], position[2] - position[0])
    sp_angle = math.degrees(myradians)
    target_distance = math.hypot(position[2] - position[0], position[3] - position[1])
    target_angle = euler[2]
    sp_angle = angle_saturation(sp_angle)
    spHeading = sp_angle
    sp_angle = -sp_angle
    target_angle = angle_saturation(target_angle)
    target_angle = -target_angle
    current_heading = math.radians(target_angle)
    err = sp_angle - target_angle
    err = angle_saturation(err)
    err = P(err) + Integral(err)
    rudder_angle = err / 2
    if err > 60:
        err = 60
    if err < -60:
        err = -60
    return rudder_angle


def sail_ctrl(global_dir):
    wind_dir = global_dir - math.degrees(current_heading)
    wind_dir = angle_saturation(wind_dir+180)
    sail_angle = math.radians(wind_dir)/2;
    if math.degrees(sail_angle) < -80:
        sail_angle = -sail_angle
    return math.degrees(-sail_angle)


def verify_result():
    global target_distance
    global f_distance
    if target_distance < f_distance:
        result = 1
    if target_distance >= f_distance:
        result = 0
    return result
