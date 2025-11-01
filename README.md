

# ROS2 foxy simulation with ackerman model car

## Docker
### 1. Build and Launch the Docker Container

```bash
cd bmw_gtr_ros2
```
```bash
# running on computer
docker compose -f docker-compose.pc.yml up -d --build
```
```bash
# running on pi
docker compose -f docker-compose.rpi.yml up -d --build
```
If the image was already build you can skip the ```--build``` flag. The flag ```-f``` is used to specify the compose file, used in our case because we have 2 of this file type.

### 2. Access the Container
```bash
# computer
docker exec -it ros2-foxy-car bash
```
```bash
# pi
docker exec -it mpc-car bash
```


## Ros2 workspace
Once inside the container we can start using the workspace. If the container was restarted by running ```docker compose ... up -d``` then we need to build the ros2 workspace again. Note! building must be done always in the folder /ros2_ws.

Building workspace
```bash
cd /ros2_ws
colcon build && source install/setup.bash

# or by running the alias "b" (check the definitioan of the alias inside the last line of code in Dockerfile.pc)
b
```

If the container was not rebuild, and just started using ```docker start CONTAINER_NAME``` then your previous build folder are not modified. You can jump straight to sourcing and running the sim. 

In new terminals you need to source ros and gazebo to interact with them. You have 2 options:
```bash
# Sourcing normal
source /ros2_ws/install/setup.bash

# or by running the alias "s" (check the definitioan of the alias inside the last line of code in Dockerfile.pc)
s
```

## Gazebo simulation / launchers
We have already defined launcher to start more easily scripts and ros nodes. To make it even easier we have aliases for those launcher.
```bash
# Launch Gazebo + world
ros2 launch simulator simulator.launch.py
# or by calling the alias
sim
```
```bash
# Remove the car from the word and spawn it
ros2 launch simulator spawn_car.launch.py
# or by calling the alias
spawn
```
```bash
# Remove the car from the word, spawn it, and launch the main_brain.py
ros2 launch simulator mpc.launch.py
# or by calling the alias
mpc
```

## #. Extra docker commands if needed 
```bash
# List containers
docker ps -a

# Stop and remove a container
docker stop rm CONTAINER_NAME

# Check images availabel
docker images

# Remove images
docker rmi IMAGENAME
```
