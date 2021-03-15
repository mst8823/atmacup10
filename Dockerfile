FROM continuumio/anaconda3:2020.11

RUN conda install -y tensorflow-gpu

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

COPY requirements.txt .
RUN pip install -U pip && \
    pip install -r requirements.txt

WORKDIR /workdir