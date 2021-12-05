import communicate as cm
import original_controller as ctr

class serial_port:
    direction=''
    port_name=''
    timeout=0
    j = False
    
    def __init__(self, direct, port, time):
        self.direction = direct
        self.port_name = port
        self.timeout = time
        
    def read_data_sensor(self):
        band = False
        [band,recibe] = cm.read_data()
        if band and self.j:
            position =  [recibe['S1'],recibe['S2']]
            angles = [recibe['S5'],recibe['S6'],recibe['S7']]
            wind = recibe['S8']
            speed = [recibe['S3'],recibe['S4']]
            band = [band]+position+angles+[wind]+speed

        return band
    
    def read_data_sensor_2(self):
        band = False
        [band,recibe] = cm.read_data()
        if band and self.j:
            position =  [recibe['S1'],recibe['S2'],recibe['S3'],recibe['S4']]
            angles = [recibe['S5'],recibe['S6'],recibe['S7']]
            wind = recibe['S8']
            band = [band]+[position]+[angles]+[wind]
            print(position)

        return band

    def write_control_action(self,control_action):
        band=False
        if self.j:
            datos={'A1' : control_action[0], 'A2' : control_action[1], 'A3' : control_action[2]}
            band=cm.write_data(datos)

        return band

    def classic_control_action(self,data,info=1000):
        if info==1000:
            rudder_angle = ctr.rudder_ctrl(data[1],data[2])
            sail_angle = ctr.sail_ctrl(data[3])
        else:
            rudder_angle = info
            sail_angle = ctr.sail_ctrl(data)
        result = ctr.verify_result()

        return [rudder_angle,sail_angle,result]

    def open_port(self):
        l=cm.serial_initialization(self.direction,self.port_name,self.timeout)
        if not l:
            print('El puerto no pudo ser abierto')
        else:
            print("Puerto abierto correctamente")
        self.j=l
        return l

    def classic_controller(self):
        while self.j:
            data=self.read_data_sensor_2()
            if not isinstance(data,bool):
                control = self.classic_control_action(data)
                self.write_control_action(control)
                #print(data[3])
            else:
                print("No hay dato")
                

