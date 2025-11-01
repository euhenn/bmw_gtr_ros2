

# ROS2 foxy simulation with ackerman model car

## Docker
### 1. Build and Launch the Docker Container

```bash
cd bmw_gtr_ros2
#docker compose up -d
docker compose -f docker-compose.pc.yml up -d

```

### 2. Access the Container
```bash
# From the same directory as docker-compose.yml
docker compose exec ros2-car bash

# From any directory (using container name)
docker exec -it ros2-car bash
```
Note, `docker compose` commands work only in the directories containing the compose file.


### #. Extra docker commands if needed 
```bash

# Stop the container
docker stop ros2-car

# Delete the container
docker rm ros2-car

# Force to rebuild
docker compose up -d --build

# Check open containers
docker ps -a

# Check images availabel
docker images

# Remove images
docker rmi IMAGENAME
```
Note, `docker compose` commands work only in the directories containing the compose file.


## Ros2 workspace

Build the project, must be done always in the folder /ros2_ws, and source the setup.

```bash
cd /ros2_ws
colcon build && source install/setup.bash
```

## Gazebo simulation
Launch gazebo simulation scenario 
```bash
ros2 launch bmw_gtr simulator.launch.py
```
if you need to remove the car from the word and spawn it at a certain coordinates use this launch file:
```bash
ros2 launch bmw_gtr spawn_car.launch.py
```

## Main brain
Source the ros project as we need in order to be able to use the ros topics.
