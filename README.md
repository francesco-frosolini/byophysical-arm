WIP

Current state: a Look Up Table can be used to get muscle activations for a requested torque. The resulting torque is pretty close to the requested one, but it is expected to place this mapping inside of a position control loop which can cancel steady state errors.

The LUT contains the 6 MyoSuite arm muscles' activations for a given torque at a given angle, which where computed using Jiang et al. 2007's Entropy-Assisted Optimization Model. In turn, the optimization procedure was performed using a LUT of torques given by muscles activating singularly. The velocity contribution of the Hill's muscolar model was ignored in this step.

If this sounds like Aramaic it's because it's my unfiltered stream of conciousness.
