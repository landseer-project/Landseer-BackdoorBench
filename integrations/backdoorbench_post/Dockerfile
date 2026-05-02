FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    torchvision \
    numpy \
    pillow \
    pyyaml \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    opencv-python-headless \
    tensorboard \
    lpips \
    kornia \
    scikit-image \
    higher \
    umap-learn

COPY main.py /app/main.py

LABEL org.opencontainers.image.title="Landseer BackdoorBench Post-Training Wrapper" \
      org.opencontainers.image.description="Runs BackdoorBench ft/sft as Landseer post-training defenses" \
      org.opencontainers.image.stage="post_training" \
      org.opencontainers.image.dataset="cifar10" \
      org.opencontainers.image.defense_type="backdoor"

ENTRYPOINT ["python", "/app/main.py"]
