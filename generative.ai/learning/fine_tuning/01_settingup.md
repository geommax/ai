## GUI Frameworks for Fine Tuning.  

| Tool | Ease of Use | Full Fine-Tuning Support? | Platform |
| --- | --- | --- | --- |
| **AutoTrain** | High (No-code) | Yes (Managed) | Hugging Face Spaces |
| **LLaMA-Factory** | Medium (Dashboard) | Yes | Local / Kaggle / Colab |
| **Unsloth UI** | Medium (Gradio) | Limited (Mostly LoRA) | Local / Colab |
| **Axolotl UI** | High (Editor) | Yes | Web / Local |

### 01. Setting UP with Axolotl in Local Lab.

```bash
https://github.com/axolotl-ai-cloud/axolotl
```

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```
### Known Issues For Above commands
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

## Fix By installing the following packages 
```bash
# 1. Configure the package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

```bash
# Configure the runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
sudo systemctl restart docker

# To verify your GPU pass-through, use this updated, guaranteed-to-exist tag for CUDA 12:
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## RUN inside container

```bash
hf auth login --token $HF_TOKEN
```
