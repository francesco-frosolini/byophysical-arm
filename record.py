import mujoco as m
import imageio
import numpy as np

def render_state(model, state):
    data = m.MjData(model)
    m.mj_setState(model, data, state, m.mjtState.mjSTATE_INTEGRATION)
    m.mj_fwdPosition(model, data)
    return data

def render_frames(model, states_buffer, height, width, camera="front_facing"):
    frames = []
    with m.Renderer(model, height, width) as r:
        replay_data = m.MjData(model)
        

        for i, state in enumerate(states_buffer):
            replay_data= render_state(model, state)

            if i==0:
                print(f"Time of the first saved data object: {replay_data.time}")
                print(f"Number of frames to render: {len(states_buffer)}")
            if i==len(states_buffer)-1:
                print(f"Time of the last saved data object: {replay_data.time}\n")
            r.update_scene(replay_data,camera)

            #TODO - manage errors, add TimeOverlay flag
            pixels=r.render()  
            if r._mjr_context is not None:
                draw_time_overlay(replay_data, r._mjr_context, width, height)
                viewport=m.MjrRect(0, 0, width, height)
                
                m.mjr_readPixels(pixels, None,viewport,r._mjr_context)
                #for some reason using this flips the image upside down, so we flip it back
            
            pixels_flipped = np.flipud(pixels)
            frames.append(pixels_flipped)
    return frames

def save_video(frames, save_name, fps):
    try:
        output_path = f"videos/{save_name}.mp4"
        imageio.mimwrite(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")

def draw_time_overlay(data, mjr_context: m.MjrContext, w, h):

    pos = m.mjtGridPos.mjGRID_TOPLEFT
    
    #whole screen viewport
    viewport = m.MjrRect(0, 0, w, h)
    
    line1 = f"ELAPSED TIME: {data.time:.3f}s"
 
    m.mjr_overlay(
        m.mjtFont.mjFONT_BIG,
        pos, 
        viewport, 
        line1, 
        "", 
        mjr_context
    )