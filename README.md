

# ROS2 foxy simulation with ackerman model car


```bash
ros2 topic pub /automobile/command std_msgs/msg/String '{"data":"{\"action\": \"1\", \"speed\":0.0}"}'
```


To build the image do 
```bash
docker compose up -d
```
 inside the folder with dockerfile

```bash
docker exec -it ros2-foxy-car bash
```
 to open the container

once inside the container build the workspace with 


```bash
colcon build && source install/setup.bash
```


and run the launcher
```bash
ros2 launch bmw_gtr simulator.launch
```

