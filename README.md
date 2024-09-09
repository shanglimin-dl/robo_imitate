```sh 
docker build --build-arg UID=$(id -u) --build-arg DATA_PATH=robot_imitate/data/XXX.parquet -t robot_imitate .
```

```sh
docker run -v /home/marija/test_docker/robot_imitate/:/docker/app/robot_imitate --gpus all -it  robot_imitate
```
