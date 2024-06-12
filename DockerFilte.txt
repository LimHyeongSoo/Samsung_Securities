# base image
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# timezone 설정 : Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8000
EXPOSE 8000

# apt-get update and install necessary packages
RUN apt-get update \
    && apt-get install -y wget git vim g++ gcc make curl locales \
    && rm -rf /var/lib/apt/lists/*

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
ENV PATH="/root/miniconda/bin:${PATH}"

# create conda env
RUN conda create -n samsung2 python=3.10 -y && conda init bash

# activate conda env
RUN echo "conda activate samsung2" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# set working directory
WORKDIR /samsung-digital-it-team2

# copy requirements file and install packages
COPY requirements.txt .
RUN /root/miniconda/envs/samsung2/bin/pip install -r requirements.txt

# # copy model download script and run it
# COPY model_download.py .
# RUN /root/miniconda/envs/samsung2/bin/python model_download.py

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# copy project files
COPY . .

# copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# command to run the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]


# docker build -t final-chat-app .
# docker run -it --name final-container -p 8000:8000 final-chat-app
