import mujoco as m
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import record

IMG_HEIGHT = 1088
IMG_WIDTH = 1920
SIMLEN = 4 #seconds
DPI=150
FPS=30
TIMESTEP=0.001


def save_screenshot(pixels, filename):
    try:
        img = Image.fromarray(pixels)
        img.save(f"screenshots/{filename}")
        print(f"Screenshot saved to screenshots/{filename}")
    except Exception as e:
        print(f"Error saving screenshot: {e}")

def main():
    try:
        model = m.MjModel.from_xml_path("robotic_arm.xml")
        model.opt.timestep = TIMESTEP
        data = m.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    #load salute keyframe
    salute_k=model.key('salute')
    m.mj_resetDataKeyframe(model, data,salute_k.id)


    with m.Renderer(model, IMG_HEIGHT, IMG_WIDTH) as r:
        # Render the initial state
        m.mj_forward(model, data)
        r.update_scene(data,camera="front_facing")

        pixels = r.render()
        save_screenshot(pixels, "initial_state.png")

    hand_x = data.body('hand').xpos

    #sh_j1=data.joint('shoulder_rotation2').qpos[0]

    print("hand x pos: ",hand_x,"\n")


    time_series = []  # list of (time, hand_z) tuples
    states = []
    timestep = 0

    state_size = m.mj_stateSize(model, m.mjtState.mjSTATE_INTEGRATION)

    #run sim
    simstart=data.time
    while (data.time-simstart) < SIMLEN:
        
        if  len(states)<(data.time-simstart)*FPS:
            print(data.time-simstart, "seconds simulated\n")

            state_buffer = np.empty(state_size, dtype=np.float64)
            m.mj_getState(model, data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
            states.append(state_buffer.copy())

        timestep += 1

        hand = data.body('hand')
        time_series.append((data.time, hand.xpos[2]))

        m.mj_step(model, data)
        
    print(timestep, "timesteps taken\n"
          ,len(states), "frames saved\n")

    with m.Renderer(model,IMG_HEIGHT, IMG_WIDTH) as r:
        #render final state
        m.mj_forward(model, data)
        r.update_scene(data,camera="front_facing")

        pixels = r.render()
        save_screenshot(pixels, "final_state.png")


    
    # plot using record helper
    record.plot_data(time_series, "hand_height")

    # also draw the video with live overlay
    record.save_video(
        record.render_frames(model, states, IMG_HEIGHT, IMG_WIDTH, time_series=time_series),
        "salute_video",
        FPS
    )

if __name__ == "__main__":
    main()