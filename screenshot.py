import mujoco as m
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import record

IMG_HEIGHT = 1080
IMG_WIDTH = 1920
DPI=150
FPS=60
TIMESTEP=0.002
SPS=(1/TIMESTEP)/FPS


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


    timevals = []
    hand_z0_t=[]
    states = []
    timestep = 0

    state_size = m.mj_stateSize(model, m.mjtState.mjSTATE_INTEGRATION)

    #run sim
    simstart=data.time
    while (data.time-simstart) < 5.0:
        
        if  len(states)<(data.time-simstart)*FPS:
            print(data.time-simstart, "seconds simulated\n")

            state_buffer = np.empty(state_size, dtype=np.float64)
            m.mj_getState(model, data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
            states.append(state_buffer.copy())

        timestep += 1

        timevals.append(data.time)
        hand=data.body('hand')
        hand_z0_t.append(hand.xpos[2])

        m.mj_step(model, data)
        
    print(timestep, "timesteps taken\n"
          ,len(states), "frames saved\n")

    with m.Renderer(model,IMG_HEIGHT, IMG_WIDTH) as r:
        #render final state
        m.mj_forward(model, data)
        r.update_scene(data,camera="front_facing")

        pixels = r.render()
        save_screenshot(pixels, "final_state.png")


    
    width = 600
    height = 800
    figsize = (width / DPI, height / DPI)
    _, ax = plt.subplots(figsize=figsize, dpi=DPI)

    ax.plot(timevals, hand_z0_t, label='hand height')
    ax.set_title('hand height over time\n')
    ax.set_ylabel('meters / second')


    # Save the plot

    # DIRECTORY MUST EXIST 
    plt.savefig('plots/hand_height.png')

    record.save_video(record.render_frames(model, states, IMG_HEIGHT, IMG_WIDTH), "salute_video", FPS)

if __name__ == "__main__":
    main()