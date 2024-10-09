![Screwdriver picking](./media/teleop.png)


| ![Lite6 Physical Teleoperation](./media/pick_object.gif)  | ![UR5e Webots Teleoperation](./media/move_object.gif) |
|:-------------------------------------------------------------------:|:----------------------------------------------------:|
| Object picking with imitation learning                         | Object picking when pose of object is randomized    |


### Model training

Inside `robo_imitate` directory run follow commands:

```sh 
docker build --build-arg UID=$(id -u) -t robot_imitate .
```

```sh
docker run -v $(pwd)/robot_imitate/:/docker/app/robot_imitate:Z --gpus all -it -e DATA_PATH=robot_imitate/data/2024_09_09_19_47_17.parquet -e EPOCH=10 robot_imitate
```

### Model evaluation
[!IMPORTANT]  
You need to have installed docker. If you have Nvidia GPU you need additionaly follow this [file](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

- Build docker container
```sh
make build-pc run exec
```
- Build ROS 2 packages
```sh
colcon build --symlink-install && source ./install/local_setup.bash
```
- Run ROS 2 controler
```sh
ros2 launch xarm_bringup lite6_cartesian_launch.py rviz:=false sim:=true
```
If you want to vizualize robot set `rviz` on true.

- Open anather terminal and run docker
```sh
make exec
```

- Run model inside docker
```sh
cd src/robo_imitate && ./robo_imitate/inference
```



python3 ./robot_imitate/compute_stats --path robot_imitate/data/2024_09_09_19_47_17.parquet  && python3 ./robot_imitate/train_script --path robot_imitate/data/2024_09_09_19_47_17.parquet  --epoch 1000