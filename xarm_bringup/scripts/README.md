Inside this foilder you can find diferent script for detalolection and robot teleoperation. This folder contain:

```
  .scripts/
    |-- episode_generator_picking        # Controls the simulated robot during data collection
    |-- episode_manager                  # Manages data saving in real environments
    |-- episode_recorder                 # Saves episodes during data collection
    |-- keyboard_teleop                  # Controls the robot via keyboard teleoperation
    |-- lite6_parallel_gripper_controller # Run this to open/close the gripper
    |-- save_parquet                     # Saves collected data in parquet format
    |-- sixd_speed_limiter               # Splits the path into smaller segments to prevent jerks
    |-- space_teleop                     # Controls the robot via space mouse teleoperation

```

### Usage

- **Simulated data collection**
If you want to collect data in simulation you need to run scripts:
  1. episode_generator_picking
  2. episode_recorder --data_dir FILE_NAME
  

- **Real data collection**
If you want to collect real data you need to run scripts:
  1. episode_manager
  2. episode_recorder --data_dir FILE_NAME
  3. space_teleop/keyboard_teleop

  Using episode_manager you can controll start end episode and break points in one episode.


- **Data saving**
If you want to save data in properly format you need to run:
```sh
./save_parquet --data_path DATA_PATH
```


