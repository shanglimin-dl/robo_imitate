# Imitation learning 

| ![Lite6 Physical Teleoperation](./media/pick_object.gif)  | ![UR5e Webots Teleoperation](./media/move_object.gif) |
|:-------------------------------------------------------------------:|:----------------------------------------------------:|
| Object picking with imitation learning                         | Object picking when pose of object is randomized    |

<div align="center">
	<img src="./media/robo_imitate.png">
</div>



### Model training

Inside `robo_imitate` directory run follow commands:

```sh 
docker build --build-arg UID=$(id -u) -t imitation .
```

```sh
docker run -v $(pwd)/imitation/:/docker/app/imitation:Z --gpus all -it -e DATA_PATH=imitation/data/2024_09_09_19_47_17.parquet -e EPOCH=10 imitation
```

>[!TIP]
 If you want to run model training inside docker, run this command inside the folder `src/robo_imitate`. Before that, you need to build the docker (see section "Model evaluation").. 

```sh
python3 ./imitation/compute_stats --path imitation/data/2024_09_09_19_47_17.parquet  && python3 ./imitation/train_script --path imitation/data/2024_09_09_19_47_17.parquet  --epoch 1000
```


### Model evaluation
>[!IMPORTANT]  
You need to have Docker installed. If you have an Nvidia GPU, you need to additionally follow this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). You need to istall Isaac-Sim.

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
cd src/robo_imitate && ./imitation/inference
```