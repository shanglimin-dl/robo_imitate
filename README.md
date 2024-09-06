```docker build --build-arg UID=$(id -u) -t robot_imitate .```

`docker run -v /home/marija/test_docker/robot_imitate/:/docker/app/robot_imitate:Z --gpus all -it  robot_imitate`
