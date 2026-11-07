# Environment Setup

This file records the Docker-based environment setup used by the benchmark
runners. Treat package, driver, CUDA, ROCm, and image versions as pinned examples;
adjust them for the target machine when needed.

Use a fresh clone of this repository as the host-side `<repo-dir>`, then mount it
at `/workspace` inside the container.

## NVIDIA Docker Setup

### Host Prerequisites

Install an NVIDIA driver, Docker, and the NVIDIA Container Toolkit on the host.
The driver package names below are examples for Ubuntu 22.04; choose a driver
version that matches the target GPU and CUDA runtime.

```bash
sudo apt update
sudo apt install -y dkms nvidia-utils-580 nvidia-driver-580
sudo reboot

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker "$USER"
# Log out and back in after adding the user to the docker group.

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Start The Container

```bash
docker run --gpus all -it \
  --shm-size=128g \
  -v <repo-dir>:/workspace \
  --name <container-name> \
  ubuntu:22.04 bash

# Later sessions:
docker start <container-name>
docker exec -it <container-name> bash
```

### Container Bootstrap

```bash
apt update
apt upgrade -y
apt install -y \
  build-essential curl git git-lfs g++-11 ninja-build unzip vim wget \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 999
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 999
```

Install CUDA 12.8 inside the container:

```bash
cd /root
export DISTNAME=ubuntu2204
wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTNAME}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-toolkit-12-8=12.8.1-1
cat <<'EOF' >> ~/.bashrc
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
source ~/.bashrc
```

Install Python 3.12.13 and `uv`:

```bash
cd /root
wget https://www.python.org/ftp/python/3.12.13/Python-3.12.13.tgz
tar -xvf Python-3.12.13.tgz
cd Python-3.12.13
./configure --enable-optimizations --with-lto
make -j"$(nproc)"
make altinstall
update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 999
python -m pip install --upgrade pip
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Install shared Python/build dependencies and initialize the repository:

```bash
uv pip install cmake setuptools_scm immutabledict langdetect matplotlib --system

git config --global --add safe.directory /workspace
cd /workspace
git submodule sync --recursive
git submodule update --init --recursive
```

Build the pinned `vllm/` checkout for CUDA with the repository helper. It uses
`TORCH_BACKEND=cu128` and `UV_SYSTEM_PYTHON=1` by default, installs the Python
editable package without fetching precompiled vLLM wheels, generates the CMake
release preset, then builds the native install target.

```bash
cd /workspace
scripts/vllm_incremental_build.sh setup --confirm-setup-or-configure
scripts/vllm_incremental_build.sh configure --confirm-setup-or-configure
scripts/vllm_incremental_build.sh build
```

For later CUDA native-code edits, rerun only the incremental build step:

```bash
cd /workspace
scripts/vllm_incremental_build.sh build
```

## AMD Docker Setup

### Host Prerequisites

Install the ROCm host driver stack for the target GPU before starting the
container. Verify that `/dev/kfd` and `/dev/dri` are available on the host and
that the user can access them through the `video` and `render` groups.

### Start The Container

The ROCm setup uses the public vLLM ROCm image as the base image. Replace
`<repo-dir>` and `<container-name>` with local names.

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --shm-size=128g \
  -v <repo-dir>:/workspace \
  --name <container-name> \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.19.0

# Later sessions:
docker start <container-name>
docker exec -it <container-name> bash
```

### Container Bootstrap

The base image already includes Python and ROCm runtime components. Remove the
preinstalled vLLM wheel so the mounted editable checkout is used instead.

```bash
pip uninstall -y vllm

apt update
apt upgrade -y
apt install -y \
  build-essential curl git git-lfs ninja-build unzip vim wget \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev

python -m pip install --upgrade pip
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv pip install cmake setuptools_scm immutabledict langdetect matplotlib --system

git config --global --add safe.directory /workspace
cd /workspace
git submodule sync --recursive
git submodule update --init --recursive
```

Build the pinned `vllm/` checkout for ROCm. Set `PYTORCH_ROCM_ARCH` for the
target GPU architecture; `gfx942` is the MI300-class example used here.

```bash
cd /workspace/vllm
uv pip install -v -e . --no-build-isolation --system

PYTORCH_ROCM_ARCH=gfx942 cmake -G Ninja -S . -B cmake-build-release \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE="$(which python)" \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$(pwd)"

PYTORCH_ROCM_ARCH=gfx942 cmake --build cmake-build-release --target install -j 32
```

## Patch Workflow

For CUDA/NVIDIA runs after the helper setup above, keep `vllm/` clean and use
`prepare-version`; it applies the requested patch and runs the incremental build:

```bash
cd /workspace
scripts/vllm_incremental_build.sh prepare-version VLLM_HEAL
# run experiment while the patch is active
scripts/vllm_incremental_build.sh revert-version
```

For ROCm/AMD runs, use the helper only to apply and revert patches, then rebuild
with the ROCm CMake tree from the AMD setup section:

```bash
cd /workspace
scripts/vllm_incremental_build.sh apply-version VLLM_HEAL
cd /workspace/vllm
PYTORCH_ROCM_ARCH=gfx942 cmake --build cmake-build-release --target install -j 32
# run experiment while the patch is active
cd /workspace
scripts/vllm_incremental_build.sh revert-version
```
