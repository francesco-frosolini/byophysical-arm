#load a position, start simulation with gravity, wait a bit, use actuators to hold position
from traceback import print_list

import mujoco as m
import numpy as np
import record


IMG_HEIGHT = 1088
IMG_WIDTH = 1920
SIMLEN = 8 #seconds
DPI=150
FPS=30
TIMESTEP=0.001


def main():
    try:
        model = m.MjModel.from_xml_path("models/robotic_arm.xml")
        spec=m.MjSpec.from_file("models/robotic_arm.xml")
        #make model from spec
        model.opt.timestep = TIMESTEP
        data = m.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    #load salute keyframe
    salute_k=model.key('salute')
    m.mj_resetDataKeyframe(model, data,salute_k.id)

    #run sim
    states = []
    state_size = m.mj_stateSize(model, m.mjtState.mjSTATE_INTEGRATION)

    simstart = data.time
    # Get sensor dimensions and initialize sensor_series
    sensord = data.sensor("sensor_sh1").data
    sensor_dim = len(sensord)
    sensor_series = [[] for _ in range(sensor_dim)]
    act = None

    while (data.time - simstart) < SIMLEN:
        # Save state every 1/FPS seconds
        if len(states) < (data.time - simstart) * FPS:
            #print(data.time-simstart, "seconds simulated\n")

            state_buffer = np.empty(state_size, dtype=np.float64)
            m.mj_getState(model, data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
            states.append(state_buffer.copy())
        # Disable gravity after 1 second (run once)
        if data.time - simstart >= 0.5 and model.opt.gravity[2] != 0:
            model.opt.gravity[2] = 0
            print(f"Gravity disabled at {data.time - simstart} seconds\n")
        
        # record sensor reading
        sensord = data.sensor("sensor_sh1").data
        for i in range(len(sensord)):
            sensor_series[i].append((data.time, sensord[i]))

        m.mj_step(model, data)
    
    #________________
    #AFTER SIMULATION

    #record.save_video(record.render_frames(model, states, IMG_HEIGHT, IMG_WIDTH), "hold_salute", FPS)
    for i in range(sensor_dim):
        record.plot_data(sensor_series[i], f"sensor_sh1_{i}")



if __name__ == "__main__":
    main()