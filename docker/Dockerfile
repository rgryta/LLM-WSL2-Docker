################ Download models ################
FROM alpine:latest AS dw_models
RUN apk add wget curl jq

RUN mkdir -p /llm/models

### MODELS:
# WizardLM
RUN mkdir /llm/models/WizardLM
WORKDIR /llm/models/WizardLM
	
RUN for file in `curl https://huggingface.co/api/models/ehartford/WizardLM-7B-V1.0-Uncensored | jq -r .siblings[].rfilename`; \
	do curl -L -C - https://huggingface.co/ehartford/WizardLM-7B-V1.0-Uncensored/resolve/main/$file --create-dirs -o $file; done

# Replace with this to use 13B model (or a different one)
# RUN for file in `curl https://huggingface.co/api/models/ehartford/WizardLM-13B-V1.0-Uncensored | jq -r .siblings[].rfilename`; \
#	do curl -L -C - https://huggingface.co/ehartford/WizardLM-13B-V1.0-Uncensored/resolve/main/$file --create-dirs -o $file; done



################ Main container ################
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as llm_webui

ENV LLM_HOME=/home/llm
ENV LLM_PROJECT=$LLM_HOME/text-generation-webui

# Install necessary packages
RUN apt-get update && \
	apt-get install --no-install-recommends --no-install-suggests -y \
		sudo git g++ python3 python3-pip python3-venv python3-dev ffmpeg libsm6 libxext6 curl wget vim

# Create LLM user
RUN useradd -m -G sudo -p $(openssl passwd -1 llm) llm
USER llm

# Python setup - PYTHONUNBUFFERED set to 1 makes logs pipe into the container output
RUN pip install --upgrade pip
# Torch support list based on https://github.com/pytorch/builder/blob/main/conda/pytorch-nightly/build.sh
# and https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
ENV PYTHONUNBUFFERED=1 \
	PATH="$PATH:$LLM_HOME/.local/bin" \
	TORCH_CUDA_ARCH_LIST=All \
	FORCE_CUDA=1

RUN pip install torch torchvision torchaudio xformers

# Clone text generation webui repo
RUN cd $LLM_HOME && git clone https://github.com/oobabooga/text-generation-webui.git

WORKDIR $LLM_PROJECT
RUN pip install -r requirements.txt


# Copy models
COPY --from=dw_models --chown=llm /llm/ $LLM_PROJECT/

EXPOSE 7860/tcp
CMD ["/bin/bash", "-c", "python3 server.py --chat --auto-devices --xformers --listen --load-in-4bit"]