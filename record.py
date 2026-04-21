import mujoco as m
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

DPI=150
PLOT_W=800
PLOT_H=600
LIVEPLOT_RESIZE_FACTOR=2 #INTEGER
LIVEPLOT_W_FRAC=0.4 
LIVEPLOT_Y_FRAC=0.4

def render_state(model, state):
    data = m.MjData(model)
    m.mj_setState(model, data, state, m.mjtState.mjSTATE_INTEGRATION)
    m.mj_fwdPosition(model, data)
    return data

def time_liveplot(time_series: list[tuple[float, float]], current_time: float, width, height, dpi=100, plot_title: str | None = "time_series"):
    """Generate a matplotlib plot as a numpy image array from a time-series list of (t,y) tuples.
    The plot x axis is fixed to the whole length of the time series but only data up to current_time is plotted."""
    # unzip series
    title = plot_title
    timevals, plot_y_data = zip(*time_series)
    timevals = np.array(timevals)
    plot_y_data = np.array(plot_y_data)

    fullsize_fig = plt.figure(figsize=(width*LIVEPLOT_RESIZE_FACTOR/dpi, height*LIVEPLOT_RESIZE_FACTOR/dpi), dpi=dpi/2)  #create larger figure for better spacing then subsamle
    ax = fullsize_fig.add_subplot(111)
    
    # Filter data up to current time
    mask = timevals <= current_time
    timevals_filtered = timevals[mask]
    plot_y_data_filtered = plot_y_data[mask]
    
    # Use final time + 10% margin for fixed x-axis
    max_time = timevals[-1] * 1.1
    
    # Calculate y-axis limits with ±10% margin
    y_min = np.min(plot_y_data)
    y_max = np.max(plot_y_data)
    y_range = y_max - y_min
    y_margin = y_range * 0.1
    y_min_limit = y_min - y_margin
    y_max_limit = y_max + y_margin
    
    ax.plot(timevals_filtered, plot_y_data_filtered, 'b-', linewidth=2)
    ax.set_xlim(0, max_time)
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.set_xlabel('Time (s)')
    if title is not None: ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Convert plot to image (numpy array)
    canvas = FigureCanvasAgg(fullsize_fig)
    canvas.draw()

    rgba_array = np.asarray(canvas.buffer_rgba())
    fullsize_image = rgba_array[:, :, :3] #remove alpha channel
    image=downsample_nx(fullsize_image, LIVEPLOT_RESIZE_FACTOR)

    plt.close(fullsize_fig)
    return image
        

def overlay_image(frame, image_to_overlay):
    """Overlay an image on the bottom-left of the frame with padding and a two pixel black border"""
    h_frame, w_frame = frame.shape[:2]
    h_plot, w_plot = image_to_overlay.shape[:2]
    
    padding = 10
    y_start = h_frame - h_plot - padding
    x_start = padding
    
    y_end = min(y_start + h_plot, h_frame)
    x_end = min(x_start + w_plot, w_frame)
    y_plot_end = y_end - y_start
    x_plot_end = x_end - x_start
    
    # Overlay
    frame[y_start:y_end, x_start:x_end] = image_to_overlay[:y_plot_end, :x_plot_end]
    
    # Add black border
    frame[y_start-2:y_start, x_start:x_end] = 0  # Top border
    frame[y_end:y_end+2, x_start:x_end] = 0  # Bottom border
    frame[y_start:y_end, x_start-2:x_start] = 0  # Left border
    frame[y_start:y_end, x_end:x_end+2] = 0  # Right border

    return frame

def render_frames(model,states_buffer, height, width, camera=None, time_series=None, plot_title=None):
    """Recreate frames from a buffer of states.

    :param time_series: optional list of (time, y) tuples that will be plotted over the
                        replay. If provided, it is used to generate a live plot overlay.
    """
    frames = []
    with m.Renderer(model, height, width) as r:
        replay_data = m.MjData(model)
        
        for i, state in enumerate(states_buffer):
            replay_data = render_state(model, state)

            if i == 0:
                print(f"Time of the first saved data object: {replay_data.time}")
                print(f"Number of frames to render: {len(states_buffer)}")
            if i == len(states_buffer) - 1:
                print(f"Time of the last saved data object: {replay_data.time}\n")
            r.update_scene(replay_data, camera)

            pixels = r.render()
            if r._mjr_context is not None:
                draw_time_overlay(replay_data, r._mjr_context, width, height)
                viewport = m.MjrRect(0, 0, width, height)
                m.mjr_readPixels(pixels, None, viewport, r._mjr_context)

            # Flip the image vertically (OpenGL to standard image format)
            pixels_flipped = np.flipud(pixels)

            # Add live plot if data provided
            if time_series is not None:
                plot_image = time_liveplot(time_series, replay_data.time, width*LIVEPLOT_W_FRAC, height*LIVEPLOT_Y_FRAC, dpi=100, plot_title=plot_title) 
                pixels_flipped = overlay_image(pixels_flipped, plot_image)

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

def plot_data(time_series, save_name,title:None|str=None, ref_series=None):
    """Plot and save data given as a list of (time, value) tuples. Size is in record.py's globals
        If a title is not specified, it defaults to the save_name. The plot is saved in the 'plots' directory with the name save_name.png. The directory must exist (TODO created in dockerfile).
    """
    print("Plotting data\n")
    timevals, plot_y_data = zip(*time_series)
    figsize = (PLOT_W / DPI, PLOT_H / DPI)
    _, ax = plt.subplots(figsize=figsize, dpi=DPI)

    ax.plot(timevals, plot_y_data)
    if ref_series is not None:
        ref_timevals, ref_y_data = zip(*ref_series)
        ax.plot(ref_timevals, ref_y_data, 'r--', label='Reference')
        ax.legend()
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f'{save_name} over time\n')

    # Save the plot
    directory = "plots/" + os.path.dirname(save_name)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'plots/{save_name}.png')


def downsample_nx(image,n):
    # Get current dimensions
    h : np.integer
    w : np.integer
    h, w, c = image.shape
    
    # 1. Ensure dimensions are divisible by n by cropping slightly if needed
    new_h = h - (h % n)
    new_w = w - (w % n)
    cropped = image[:new_h, :new_w, :]
    
    reshaped = cropped.reshape(new_h // n, n, new_w // n, n, c)
    return reshaped.mean(axis=(1, 3)).astype(np.uint8)
