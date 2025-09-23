# in order to be able to use this script install:
# pip install docker-run-cli
docker-run \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--ipc=host \
--privileged \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-it \
--volume $PWD:/docker-ros/ws/src/target \
tillbeemelmanns/ros2-moge-trt:latest-dev \
bash


# For debugging 
# apt update && apt install -y \
# ros-jazzy-rviz2 \
# ros-jazzy-rosbag2 \
# ros-jazzy-rosbag2-storage-mcap
