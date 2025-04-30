# ===== base image with the right CUDA for torch-2.5.1 =====
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

# ---------- system packages ----------
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential git wget unzip ca-certificates sudo && \
    rm -rf /var/lib/apt/lists/*

# ---------- miniconda ----------
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

SHELL ["/bin/bash", "-c"]      # let every RUN line be bash

# ---------- create the 'universe' conda env ----------
RUN conda create -y -n universe python=3.11.11 && \
    echo "source activate universe" >> /etc/bash.bashrc

# copy requirements early to leverage Docker layer cache
COPY models/universe/univ_requirements.txt /tmp/univ_requirements.txt

# ---------- install Python & conda packages ----------
RUN source activate universe && \
    # core packages exactly as in setup_simple.sh
    pip install --no-cache-dir torch==2.5.1 torchvision torchaudio==2.5.1 && \
    conda install -y -c conda-forge gmpy2 numexpr montreal-forced-aligner && \
    conda install -y nvidia::cuda-nvcc && \
    pip install --no-cache-dir -r /tmp/univ_requirements.txt && \
    pip install --no-cache-dir onnxruntime textgrid && \
    conda clean -afy

# ---------- project source ----------
WORKDIR /workspace
COPY . /workspace
RUN git clone https://github.com/ajaybati/miipher2.0.git _miipher

# install universe package in editable mode
RUN source activate universe && pip install --no-deps -e models/universe

# ---------- default entry ----------
# keeps the 'universe' env active for any command you pass
ENTRYPOINT ["/bin/bash", "-c", "source activate universe && exec \"$@\""]

COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]

