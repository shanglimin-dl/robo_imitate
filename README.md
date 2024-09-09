```sh 
docker build --build-arg UID=$(id -u) -t robot_imitate .
```

```sh
docker run -v /home/marija/robo_imitate/robot_imitate/:/docker/app/robot_imitate:Z --gpus all -it -e DATA_PATH=robot_imitate/data/2024_08_21_20_10_53.parquet -e EPOCH=10 robot_imitate
```
