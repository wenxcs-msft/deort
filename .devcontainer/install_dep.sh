#!/bin/bash
sudo apt install curl git -y
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b
rm Mambaforge-$(uname)-$(uname -m).sh
/home/vscode/mambaforge/bin/conda run conda install onnx -y
/home/vscode/mambaforge/bin/conda run pip install nvidia-pyindex
/home/vscode/mambaforge/bin/conda run pip install onnx_graphsurgeon