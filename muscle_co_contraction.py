import __main__
import os

from requests import get

from muscle_torque_map import lut, spec_setup
import xarray as xr
from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Rendering parameters
IMG_HEIGHT = 1088
IMG_WIDTH = 1088
DPI=150
PLOT_W=800
PLOT_H=600

# Optimization parameters
CO_CONTR_W=0.4

# File paths
HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")

#LUT generation parameters
TORQUE_STEPS = 30 # Number of torque levels to sample for each angle
comb_lut = xr.open_dataarray(os.path.join(MODELS_DIR, "comb_lut_W=0.4.nc"))
func_calls=0


def main():
    #generate_lut("comb_lut_new.nc")
    #print(func_calls)
    global comb_lut
    comb_lut = xr.open_dataarray(os.path.join(MODELS_DIR, "comb_lut_W=0.4.nc"))


    plot_limits()
    
    angle = 120
    req_torque = 5.0

    plot_lut_slice(angle)
    
    activations=optimize_muscles_activation(angle, req_torque)
    activation_dict = {muscle: act for muscle, act in zip(lut.muscle.values, [round(a, 3) for a in activations])}
    print(f"Optimal activation for angle {angle} and torque {req_torque} from optimization: {activation_dict}")
    #activations=[0,0,0,1,1,1]
    print(f"Resulting torque (LUT): {total_torque(activations, angle)}")
    
    
    activations=get_activation(angle, req_torque)
    activation_dict = {muscle: act for muscle, act in zip(lut.muscle.values, [round(a, 3) for a in activations])}
    print(f"Optimal activation for angle {angle} and torque {req_torque} from combinedLUT: {activation_dict}")



def get_activation(angle, req_torque):
    mint=comb_lut.interp(angle=angle).min_torque.values
    maxt=comb_lut.interp(angle=angle).max_torque.values
    torque_norm=(req_torque - mint) / (maxt - mint)
    activations=comb_lut.interp(angle=angle, torque_norm=torque_norm).values
    return activations

def generate_lut(name):

    angles=lut.angle.values
    max_a=np.append(np.zeros(3),np.ones(3))
    min_a=np.append(np.ones(3),np.zeros(3))
    
    empty_data=np.zeros((len(angles), TORQUE_STEPS+1, len(lut.muscle)))
    empty_min_torque=np.zeros(len(angles))
    empty_max_torque=np.zeros(len(angles))
    torque_norm=np.linspace(0, 1, TORQUE_STEPS+1) #Torque is normalized between the min and max.
    #The actual torque can be obtained as: torque = torque_norm * (max_torque - min_torque) + min_torque

    comb_lut = xr.DataArray(
        data=empty_data,
        dims=["angle", "torque_norm", "activation_idx"],
        coords=dict(
            angle=(["angle"], angles),
            torque_norm=(["torque_norm"], torque_norm),
            activation_idx=(["activation_idx"], np.arange(6)),
            min_torque=(["angle"], empty_min_torque),
            max_torque=(["angle"], empty_max_torque),
        )
    )

    for angle in tqdm(angles,desc="angles"):
        mint=total_torque(min_a, angle)
        maxt=total_torque(max_a, angle)
        torques=np.linspace(mint, maxt, TORQUE_STEPS+1)
        comb_lut.coords['min_torque'].loc[angle]=mint
        comb_lut.coords['max_torque'].loc[angle]=maxt
        for (i,torque) in enumerate(tqdm(torques,desc="torques",leave=False)):
            activations=optimize_muscles_activation(angle, torque)
            #activations=np.random.rand(6)
            comb_lut.loc[angle, torque_norm[i]]=activations
    
    output_path = os.path.join(MODELS_DIR, os.path.basename(name))
    comb_lut.to_netcdf(output_path)

def plot_limits():
    import matplotlib.pyplot as plt
    plt.figure(figsize = (PLOT_W / DPI, PLOT_H / DPI), dpi=DPI)

    comb_lut.min_torque.plot(label='Min Torque', color='blue', marker='o')
    comb_lut.max_torque.plot(label='Max Torque', color='red', marker='o')

    # Add standard matplotlib embellishments
    plt.title('Torque Boundaries vs. Angle')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/muscle_torque_map.png')
    print("Plots saved to ./plots/muscle_torque_map/ folder")


def plot_lut_slice(angle_deg):
    """Plot a slice of the combined LUT using record.plot_data.

    """

    base_spec = spec_setup()
    muscle_names = [muscle.name for muscle in base_spec.actuators]
    angle_lut=comb_lut.interp(angle=angle_deg)
    act_values = [(muscle_names[i],angle_lut.sel(activation_idx=i).values) for i in range(len(muscle_names))]
    x_values = angle_lut.min_torque.values+(angle_lut.max_torque.values-angle_lut.min_torque.values)*angle_lut.torque_norm.values
    plt.figure(figsize=(10, 6))

# Iterate through the act_values list (muscle_name, activation_data)
    for muscle_name, activations in act_values:
        plt.plot(x_values, activations, label=muscle_name, linewidth=2)

    # Adding labels and title
    plt.title(f'Muscle Activations at {angle_deg}°', fontsize=14)
    plt.xlabel('Torque (Nm)', fontsize=12)
    plt.ylabel('Activation', fontsize=12)

    # Enhancing the grid and appearance
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1) # Baseline

    # Adding the legend for the 6 muscles
    plt.legend(title="Muscles", loc='best', frameon=True)
    save_name=f"{angle_deg}deg_activations"
    plt.tight_layout()
    directory = "plots/" + os.path.dirname(save_name)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'plots/{save_name}.png')
    print("Plot saved to "f"./plots/ folder")

def total_torque(activations,angle):
        torque = 0
        for i, muscle in enumerate(lut.muscle.values):
            torque += lut.sel(muscle=muscle).interp(angle=angle, activation=activations[i]).values
        return torque
def optimize_muscles_activation(angle,req_torque):
    def torque_at_angle(activations):
        global func_calls
        func_calls+=1
        return total_torque(activations, angle)
    
    def objective(activations):
        effort = (1-CO_CONTR_W)*np.sum(activations**3)
        entropy = CO_CONTR_W * np.sum(activations * np.log(activations + 1e-9))
        return effort + entropy
    constraint = {'type': 'eq', 'fun': lambda activations: torque_at_angle(activations) - req_torque}
    x0 = np.full(len(lut.muscle), 0.5)  # Initial guess # Could be made closer to make it faster
    result = minimize(objective, x0=x0, method='SLSQP', constraints=constraint, bounds=[(0, 1)]*len(lut.muscle))
    return result.x

if __name__ == "__main__":
    main()