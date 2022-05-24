FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN useradd -rm -d /home/vscode -s /bin/bash -g root -G sudo -u 1001 vscode
USER vscode
WORKDIR /home/vscode