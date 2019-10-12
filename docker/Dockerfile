FROM registry.k.avito.ru/avito/service-python/cuda/cudnn:7.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libboost-python-dev \
    libopenblas-dev \
    libjpeg-dev \
    expect \
    wget > /dev/null && \
    rm -rf /var/lib/apt/lists/*

ENV ASYNC_TEST_TIMEOUT=10 \
    # https://github.com/xianyi/OpenBLAS/wiki/faq#multi-threaded
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

COPY ./requirements.txt $PROJECT_ROOT/
RUN pip3 install --upgrade pip

COPY jupyter_notebook_config.py /root/.jupyter/

# scikit-image requires already installed numpy
RUN pip3 install --no-cache-dir numpy==1.16.2 && \
    pip3 install --no-cache-dir -r requirements.txt

RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable ExecuteTime && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable  execute_time/ExecuteTime && \
    python3.6 -m ipykernel.kernelspec

CMD ["/bin/bash"]
