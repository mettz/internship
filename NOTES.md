# Mettz's Personal Notes

## Networks

### Isaaclab Network Benchmark 

Benchmarking Isaaclab trained network (the one provided within the Isacclab repository) on the drones has shown that with the use of the ST API the inference time is around 400us which is around 300us lower than the one obtained by the learning to fly guys. So this means that the ST API is actually faster than the one used by the learning to fly guys and is of course not a bottleneck. Also, since the learning to fly guys have managed to track an eight figure trajectory with the very same network while executing the controller loop at 500hz (their CONTROL_INTERVAL_MS is set to 2ms) this means that we should have enough margin to run an even more complex network with the ST API.

### Sebastiano's Network Benchmark

I managed to benchmark also Sebastiano's network by using the appMain task. The implementation of the benchmarking code can be found under `packages/nn-bench`. In order to properly run the benchmark, you need to add the following lines to the file `vendor/crazyflie-firmware/src/config/trace.h`:
```c
extern void on_task_switched_in(void);
extern void on_task_switched_out(void);

#define traceTASK_SWITCHED_IN() on_task_switched_in()
#define traceTASK_SWITCHED_OUT() on_task_switched_out()
```
and remove the previous definition of `traceTASK_SWITCHED_IN`.
What can be clearly seen from the benchmark results is that Sebastiano's network inference time is around 6ms which is around 15 times slower than the Isaaclab network. Also, this means that Sebastiano's network cannot directly be used within the controller loop since the attitude controller should run at 500Hz while the position controller should run at 100Hz and this is clearly not possible with the current inference time of Sebastiano's network. In addition, Sebastiano's network inference time is so high that it causes other system tasks to also be delayed (if inference is executed in the stabilizer task) due to the fact that the stabilizer task runs at the highest priority and so it cannot be preempted by other tasks. This explains why the system fails to initialize properly when Sebastiano's network is used.