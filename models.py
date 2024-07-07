from simulator import DrivingSimulation, GymSimulation, MujocoSimulation
import numpy as np


class Avoid(object):
    def __init__(self):
        self.num_of_features = 4
        self.name = 'avoid'
        self.feed_size = 0
        


class Driver(DrivingSimulation):
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4
        self.name = 'driver'

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0)))

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:,0,3]-1))

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2]))

        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1]))))

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)


class Tosser(MujocoSimulation):
    def __init__(self, total_time=1000, recording_time=[200,1000]):
        super(Tosser ,self).__init__(name='tosser', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 4
        self.state_size = 5
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = [(-0.2,0.2),(-0.785,0.785),(-0.1,0.1),(-0.1,-0.07),(-1.5,1.5)]
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # horizontal range
        horizontal_range = -np.min([x[3] for x in recording])

        # maximum altitude
        maximum_altitude = np.max([x[2] for x in recording])

        # number of flips
        num_of_flips = np.sum(np.abs([recording[i][4] - recording[i-1][4] for i in range(1,len(recording))]))/(np.pi*2)
        
        # distance to closest basket (gaussian fit)
        dist_to_basket = np.exp(-3*np.linalg.norm([np.minimum(np.abs(recording[len(recording)-1][3] + 0.9), np.abs(recording[len(recording)-1][3] + 1.4)), recording[len(recording)-1][2]+0.85]))

        return [horizontal_range, maximum_altitude, num_of_flips, dist_to_basket]

    @property
    def state(self):
        return self.sim.get_state()
    @state.setter
    def state(self, value):
        self.reset()
        temp_state = self.initial_state
        temp_state.qpos[:] = value[:]
        self.initial_state = temp_state

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        arr[150:175] = [value[0:self.input_size]]*25
        arr[175:200] = [value[self.input_size:2*self.input_size]]*25
        self.ctrl = arr

    def feed(self, value):
        initial_state = value[0:self.state_size]
        ctrl_value = value[self.state_size:self.feed_size]
        self.initial_state.qpos[:] = initial_state
        self.set_ctrl(ctrl_value)
