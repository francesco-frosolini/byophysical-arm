import __main__

from requests import get

from muscle_torque_map import lut
import xarray as xr
import record
from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np

# Rendering parameters
IMG_HEIGHT = 1088
IMG_WIDTH = 1088

# Optimization parameters
CO_CONTR_W=0.2

#LUT generation parameters
TORQUE_STEPS =10 # Number of torque levels to sample for each angle
comb_lut = xr.open_dataarray("comb_lut1.nc")


def main():
    
    angle = 37
    req_torque = 5.0
    
    activations=optimize_muscles_activation(angle, req_torque)
    activation_dict = {muscle: act for muscle, act in zip(lut.muscle.values, [round(a, 3) for a in activations])}
    print(f"Optimal activation for angle {angle} and torque {req_torque} from optimization: {activation_dict}")
    #activations=[0,0,0,1,1,1]
    print(f"Resulting torque (LUT): {total_torque(activations, angle)}")
    
    #generate_lut("comb_lut1.nc")
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
    
    comb_lut.to_netcdf(name)


    
def total_torque(activations,angle):
        torque = 0
        for i, muscle in enumerate(lut.muscle.values):
            torque += lut.sel(muscle=muscle).interp(angle=angle, activation=activations[i]).values
        return torque
def optimize_muscles_activation(angle,req_torque):
    def torque_at_angle(activations):
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