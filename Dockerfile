# Use the official Ubuntu Focal image as base (ROS Foxy requirement)
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_SG.UTF-8 \
    LC_ALL=en_SG.UTF-8

# Configure locale
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_SG en_SG.UTF-8 && \
    update-locale LC_ALL=en_SG.UTF-8 LANG=en_SG.UTF-8

# Install base dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    python3-argcomplete \
    software-properties-common

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Foxy and colcon
RUN apt-get update && \
    apt-get install -y \
    ros-foxy-desktop \
    python3-colcon-common-extensions \
    ros-foxy-gazebo-ros-pkgs \
    ros-foxy-joint-state-publisher \
    ros-foxy-vision-msgs &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages globally
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# A package of small utilities that use the proc file-system. Used for killall function that kills all processes with a given name.
RUN apt-get update && \
    apt-get install -y psmisc && \
    rm -rf /var/lib/apt/lists/*

# Setup environment
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /usr/share/gazebo/setup.sh" >> ~/.bashrc && \
    echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc && \
    echo "export _colcon_cd_root=~/ros2_install" >> ~/.bashrc

WORKDIR /ros2_ws

RUN apt-get update && \
    apt-get install -y \
    bluez \
    bluez-tools \
    python3-evdev \
    python3-pyudev &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*


RUN echo 'alias dei_sim="clear && colcon build && source install/setup.bash && ros2 launch launches dei_launch.py"' >> ~/.bashrc && \
    echo 'alias build_and_source="clear && colcon build && source install/setup.bash"' >> ~/.bashrc && \
    echo 'alias keyboard="clear && source install/setup.bash && ros2 run autocar_nav keyboard_control.py"' >> ~/.bashrc
    
# CASADI AND ACADOS    
WORKDIR /

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install casadi numpy

# Clone and build acados
RUN git clone https://github.com/acados/acados.git /opt/acados \
    && cd /opt/acados \
    && git submodule update --init --recursive \
    && mkdir -p build \
    && cd build \
    && cmake -DACADOS_WITH_QPOASES=ON .. \
    && make install -j4

# Set environment variables
ENV ACADOS_SOURCE_DIR=/opt/acados
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Install Python interface
RUN pip3 install -e ${ACADOS_SOURCE_DIR}/interfaces/acados_template
    
RUN pip3 install networkx pyclothoids

WORKDIR /ros2_ws
    
CMD ["bash"]
