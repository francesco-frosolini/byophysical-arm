import mujoco as m
import imageio

def render_state(model, data, state):
    m.mj_setState(model, data, state, m.mjtState.mjSTATE_INTEGRATION)
    m.mj_fwdPosition(model, data)

def render_frames(model, states_buffer, height, width):
    frames = []
    with m.Renderer(model, height, width) as r:
        replay_data = m.MjData(model)
        
        for i, state in enumerate(states_buffer):
            render_state(model, replay_data, state)

            if i==0:
                print(f"Time of the first saved data object: {replay_data.time}")
                print(f"Number of frames to render: {len(states_buffer)}")
            if i==len(states_buffer)-1:
                print(f"Time of the last saved data object: {replay_data.time}\n")
            r.update_scene(replay_data,camera="front_facing")
            pixels = r.render()
            frames.append(pixels)
    return frames

def save_video(frames, save_name, fps):
    try:
        output_path = f"videos/{save_name}.mp4"
        imageio.mimwrite(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")

