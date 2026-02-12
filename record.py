import mujoco as m
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

def render_state(model, state):
    data = m.MjData(model)
    m.mj_setState(model, data, state, m.mjtState.mjSTATE_INTEGRATION)
    m.mj_fwdPosition(model, data)
    return data

def plot_to_image(timevals, plot_y_data, current_time, width=400, height=300, dpi=100):
    """Generate a matplotlib plot as a numpy image array."""
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Filter data up to current time
    mask = np.array(timevals) <= current_time
    timevals_filtered = np.array(timevals)[mask]
    plot_y_data_filtered = np.array(plot_y_data)[mask]
    
    # Use final time + 10% margin for fixed x-axis
    max_time = timevals[-1] * 1.1
    
    # Calculate y-axis limits with ±10% margin
    y_data_array = np.array(plot_y_data)
    y_min = np.min(y_data_array)
    y_max = np.max(y_data_array)
    y_range = y_max - y_min
    y_margin = y_range * 0.1
    y_min_limit = y_min - y_margin
    y_max_limit = y_max + y_margin
    
    ax.plot(timevals_filtered, plot_y_data_filtered, 'b-', linewidth=2)
    ax.set_xlim(0, max_time)
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Hand Height')
    ax.grid(True, alpha=0.3)
    
    # Convert plot to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    
    image = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[1], size[0], 3)
    plt.close(fig)
    
    return image

def overlay_plot(frame, plot_image):
    """Overlay a plot image on the bottom-right of the frame."""
    h_frame, w_frame = frame.shape[:2]
    h_plot, w_plot = plot_image.shape[:2]
    
    # Position at bottom-right with padding
    padding = 10
    y_start = h_frame - h_plot - padding
    x_start = w_frame - w_plot - padding
    
    y_end = min(y_start + h_plot, h_frame)
    x_end = min(x_start + w_plot, w_frame)
    y_plot_end = y_end - y_start
    x_plot_end = x_end - x_start
    
    # Overlay
    frame[y_start:y_end, x_start:x_end] = plot_image[:y_plot_end, :x_plot_end]
    
    return frame

def render_frames(model, states_buffer, height, width, camera="front_facing", timevals=None, plot_y_data=None):
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
            r.update_scene(replay_data, camera)

            pixels=r.render()  
            if r._mjr_context is not None:
                draw_time_overlay(replay_data, r._mjr_context, width, height)
                viewport=m.MjrRect(0, 0, width, height)
                
                m.mjr_readPixels(pixels, None, viewport, r._mjr_context)
            
            # Flip the image vertically (OpenGL to standard image format)
            pixels_flipped = np.flipud(pixels)
            
            # Add live plot if data provided
            if timevals is not None and plot_y_data is not None:
                plot_image = plot_to_image(timevals, plot_y_data, replay_data.time)
                pixels_flipped = overlay_plot(pixels_flipped, plot_image)
            
            frames.append(pixels_flipped)
    return frames

def save_video(frames, save_name, fps):
    # Append final frame for 2 seconds to allow reading the plot
    final_frame = frames[-1]
    extended_frames = frames + [final_frame] * 2*fps
    
    try:
        output_path = f"videos/{save_name}.mp4"
        imageio.mimwrite(output_path, extended_frames, fps=fps)
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