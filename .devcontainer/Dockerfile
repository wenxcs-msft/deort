FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install sudo -y
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN useradd -rm -d /home/vscode -s /bin/bash -g root -G sudo -u 1001 vscode
RUN chown -R vscode:sudo /opt
USER vscode
WORKDIR /home/vscode
