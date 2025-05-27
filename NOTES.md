# Mettz's Personal Notes

## Networks

- Benchmarking Isaaclab trained network (the one provided within the Isacclab repository) on the drones has shown that with the use of the ST API the inference time is around 400us which is around 300us lower than the one obtained by the learning to fly guys. So this means that the ST API is actually faster than the one used by the learning to fly guys and is of course not a bottleneck. Also, since the learning to fly guys have managed to track an eight figure trajectory with the very same network while executing the controller loop at 500hz (their CONTROL_INTERVAL_MS is set to 2ms) this means that we should have enough margin to run an even more complex network with the ST API.