import mujoco as m
import numpy as np
from PIL import Image

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
        data = m.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    #load salute keyframe
    salute_k=model.key('salute')
    m.mj_resetDataKeyframe(model, data,salute_k.id)


    with m.Renderer(model,1080,1920) as r:
        # Render the initial state
        m.mj_forward(model, data)
        r.update_scene(data,camera="front_facing")

        pixels = r.render()
        save_screenshot(pixels, "initial_state.png")

    hand_x = data.body('hand').xpos

    #sh_j1=data.joint('shoulder_rotation2').qpos[0]

    print("hand x pos: ",hand_x,"\n")

    #run sim
    while data.time < 10.0:
        m.mj_step(model, data)


    with m.Renderer(model,1080,1920) as r:
        #render final state
        m.mj_forward(model, data)
        r.update_scene(data,camera="front_facing")

        pixels = r.render()
        save_screenshot(pixels, "final_state.png")


if __name__ == "__main__":
    main()