"""Muscle Torque Mapping Module

Locks the elbow joint at specified angles and measures the torque exerted by
individual muscles at different activation levels. Generates a lookup table (LUT)
for muscle torque as a function of joint angle and activation level.
"""

import __main__
import sys
import mujoco as m
import numpy as np
import xarray as xr
from tqdm import tqdm

sys.modules['numpy'] = np

import record

# Rendering parameters
IMG_HEIGHT = 1088
IMG_WIDTH = 1088

# Simulation parameters
SIMLEN = 0.3  # Simulation duration in seconds
FPS = 30
TIMESTEP = 0.001  # Simulation timestep in seconds

# LUT generation parameters
ACT_STEPS = 1  # Number of activation levels to sample
ANGLE_STEPS = 36  # Number of joint angles to sample (0-180 degrees)
lut = xr.open_dataarray("forward_lut.nc")


def main():

    generate_lut("forward_lut.nc")
    global lut
    lut = xr.open_dataarray("forward_lut.nc")
    # Run sample experiment with plotting
    #t = torque_experiment("BRA", angle_deg=90, activation=1, plot=True)
    #print(t)
    #t = torque_experiment("B", angle_deg=90, activation=0.5, plot=True)
    #print(t)

    spec0= spec_setup()
    muscle_names= [muscle.name for muscle in spec0.actuators]
    print("Muscles in the model:", muscle_names)
    #plot_lut_slice("BRA", angle_deg=40)
    
    for i in muscle_names:
        print(f"Plotting LUT slice for {i} at 90 degrees")
        plot_lut_slice(i,angle_deg=20)
    """
    req_torque = 8
    for muscle in muscle_names:
        t=get_activation(muscle, 90, req_torque)
        print(f"Activation for {muscle} at 20 degrees to achieve {req_torque} Nm torque: {t:.3f}")
    """

    torques=[]
    activations=[0.8, 0, 1, 0, 0, 0.4]
    angle=90

    for i,muscle in enumerate(muscle_names):
        t=lut.sel(muscle=muscle).interp(angle=angle, activation=activations[i]).item()
        torques.append(t)
    print("Torques summed from LUT: ", sum(torques))
    
    t = torque_sum_experiment(muscle_names=muscle_names, angle_deg=angle, activations=activations, plot=True)
    print("Torque from direct simulation: ", t)


def generate_lut(name):
    # Initialize lookup table (LUT) for all muscles
    base_spec = spec_setup()
    muscle_names = [muscle.name for muscle in base_spec.actuators]

    # Create angle and activation ranges for sampling
    angles = np.linspace(0, 180, ANGLE_STEPS + 1)
    activations = np.linspace(0, 1, ACT_STEPS + 1)

    # Create empty data array with proper dimensions
    data = np.zeros((len(muscle_names), len(angles), len(activations)))

    # Create xarray DataArray for organized LUT storage
    lut = xr.DataArray(
        data,
        coords=[muscle_names, angles, activations],
        dims=['muscle', 'angle', 'activation'],
        name='torque'
    )

    for muscle in tqdm(muscle_names, desc="muscles"):
        muscle_spec = setup_muscle(muscle)
        for angle in tqdm(angles, desc="angles", leave=False):
            spec_angle = muscle_spec.copy()
            set_locked_angle(spec_angle, angle)
            for activation in tqdm(activations, desc="activations", leave=False):
                torque = run_activation_experiment(spec_angle, [muscle], [activation])
                lut.loc[muscle, angle, activation] = torque

    # save LUT to disk using xarray's built-in NetCDF format
    lut.to_netcdf(name)


def setup_muscle(muscle_name):
    """Load a fresh base spec and keep only the requested muscle."""
    spec = spec_setup()

    for muscle in spec.actuators:
        muscle: m.MjsActuator
        if muscle.name != muscle_name:
            muscle.delete()  # type: ignore

    return spec


def spec_setup():
    """Load the musculoskeletal model and configure the base spec.
        No gravity and a weld constraint to lock the joint.
    """
    spec = m.MjSpec.from_file("models/myo_weld/myoelbow_0dof6muscles.xml")
    spec.option.timestep = TIMESTEP
    spec.option.gravity[2] = 0.0  # Disable gravity for pure torque measurement

    weld = spec.add_equality()
    weld.type = m.mjtEq.mjEQ_WELD
    weld.name = "elbow_weld"
    weld.name1 = "forearm"
    weld.name2 = "arm_rotation"
    weld.objtype = m.mjtObj.mjOBJ_SITE
    return spec


def torque_experiment(muscle_name, angle_deg, activation, plot=False):
    """Measure torque for a single muscle at a fixed joint angle.

    
    This convenience wrapper combines muscle setup, angle setup, and the
    final activation experiment.
    """
    spec = setup_muscle(muscle_name) # remove all muscles except the one we're testing
    set_locked_angle(spec, angle_deg)

    return run_activation_experiment(spec, [muscle_name], [activation], angle_deg=angle_deg, plot=plot)

def torque_sum_experiment(muscle_names, angle_deg, activations, plot=False):
    """Measure torque for multiple muscles activated simultaneously at a fixed joint angle. Other muscles will have 0 activation, but will still be present in the model.

    This convenience wrapper combines muscle setup, angle setup, and the
    final activation experiment for multiple muscles acting together.
    IMPORTANT: muscle_names should be a list of muscle names, and activations should be a list of corresponding activation levels
    """
    spec = spec_setup() #all muscles present
    set_locked_angle(spec, angle_deg)
    return run_activation_experiment(spec, muscle_names, activations, angle_deg=angle_deg, plot=plot)

def set_locked_angle(spec, angle_deg):
    """Update the model spec to lock the elbow at the requested angle."""
    elbow_axis = spec.joint("r_elbow_flex").axis
    spec.site("arm_rotation").delete()  # type: ignore
    spec.body("r_ulna_radius_hand").add_site(
        name="arm_rotation",
        pos=[0, 0, 0],
        axisangle=np.append(elbow_axis, -np.deg2rad(angle_deg))
    )
    return spec


def run_activation_experiment(spec : m.MjSpec, muscle_names, activations, angle_deg=None, plot=False):
    """Compile the spec and run the activation experiment to measure torque.
    IMPORTANT: muscle_names should be a list of muscle names, and activations should be a list of corresponding activation levels
    Even if only one muscle is being tested, they should still be provided as lists (e.g. muscle_names=["BRA"], activations=[0.1]) for consistency.
    """
    # Compile the modified model
    model: m.MjModel = spec.compile()
    """
        # Optional: save the modified model for inspection
    if len(spec.actuators) == 1:
        spec.to_file(f"models/myo_weld/{muscle_names[0]}_{angle_deg}deg.xml")
    else:
        spec.to_file(f"models/myo_weld/{angle_deg}deg_multiple_muscles.xml")
    """
    # Get reference to the target muscle actuator

    muscles = [model.actuator(name) for name in muscle_names]

    data: m.MjData = m.MjData(model)
    simstart = data.time

    elbow_angle_series = []
    torque_series = []
    muscles_activation_series = [[] for _ in muscles]
    total_torque = None

    while (data.time - simstart) < SIMLEN:
        for muscle in muscles:
            data.ctrl[muscle.id] = activations[muscle_names.index(muscle.name)]
        total_torque= data.qfrc_actuator[0].copy()+data.qfrc_passive[0].copy()
        if plot:
            elbow_angle_series.append((data.time, np.rad2deg(data.qpos[0].copy())))
            torque_series.append((data.time, total_torque))
            for muscle in muscles:
                muscles_activation_series[muscles.index(muscle)].append((data.time, data.ctrl[muscle.id].copy()))

        m.mj_step(model, data)

    final_torque = total_torque

    
    if plot and angle_deg is not None:
        for muscle in muscles:
            muscle_activation_series=muscles_activation_series[muscles.index(muscle)]
            record.plot_data(
                elbow_angle_series,
                f"muscle_torque_map/{angle_deg}deg/elbow_angle",
                title=f"Elbow Angle [degrees] at {angle_deg}°"
            )
            record.plot_data(
                muscle_activation_series,
                f"muscle_torque_map/{angle_deg}deg/{muscle.name}_activation",
                title=f"{muscle.name} Muscle Activation at {angle_deg}°"
            )
            record.plot_data(
                torque_series,
                f"muscle_torque_map/{angle_deg}deg/total_torque",
                title=f"Total Torque [Nm] at {angle_deg}°"
            )
        print("Plots saved to "f"./plots/muscle_torque_map/{angle_deg}deg/ folder")

    return final_torque

def get_activation(muscle_name, angle_val, target_torque):
    """Interpolate within the LUT to find the activation level that achieves the target torque."""

    # Extract the relevant slice of the LUT for the specified muscle and angle
    if angle_val<0:
        angle_val=0
    if angle_val>180:
        angle_val=180
    subset = lut.sel(muscle=muscle_name).interp(angle=angle_val)

    flexor=subset[1]>subset[0] # positive torque increment = flexor muscle

    subset_with_torque=subset.assign_coords(torque=("activation",subset.values)).swap_dims({"activation": "torque"})

    required_activation = subset_with_torque.interp(torque=target_torque).activation.item()

    if np.isnan(required_activation):
        if flexor:
            if target_torque > subset.max().item():
                required_activation = 1.0
            if target_torque < subset.min().item():
                required_activation = 0.0
        else:
            if target_torque < subset.min().item():
                required_activation = 1.0
            if target_torque > subset.max().item():
                required_activation = 0.0
    return required_activation

def plot_lut_slice(muscle_name, angle_deg=None, activation=None, torque=None):
    """Plot a slice of the torque LUT using record.plot_data.

    If angle_deg is provided, plot torque vs activation for that muscle and angle.
    If activation is provided, plot torque vs angle for that muscle and activation.
    """

    xy_series = None
    save_name = None
    title = None
    if angle_deg is None and activation is None and torque is None:
        raise ValueError("Provide exactly one of angle_deg, activation, or torque to specify the slice to plot.")

    if angle_deg is not None:
        torque_values = lut.loc[muscle_name, angle_deg, :].values
        x_values = lut.coords["activation"].values
        xy_series = list(zip(x_values.tolist(), torque_values.tolist()))
        save_name = f"muscle_torque_map/{muscle_name}/{angle_deg}deg_vs_activation"
        title = f"{muscle_name} torque vs activation at {angle_deg}°"
    elif activation is not None:
        torque_values = lut.loc[muscle_name, :, activation].values
        x_values = lut.coords["angle"].values
        xy_series = list(zip(x_values.tolist(), torque_values.tolist()))
        save_name = f"muscle_torque_map/{muscle_name}/activation_{activation:.2f}_vs_angle"
        title = f"{muscle_name} torque vs angle at activation {activation:.2f}"
    record.plot_data(xy_series, save_name, title=title)
    print("Plot saved to "f"./plots/muscle_torque_map/{muscle_name}/ folder")

if __name__ == "__main__":
    main()
