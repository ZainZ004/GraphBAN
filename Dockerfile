FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements.txt to take advantage of Docker cache
COPY requirements.txt .
RUN conda create -n graphban python=3.11 \
    && conda run -n graphban pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118  -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html\
    && conda clean --all -f -y \
    && conda run -n graphban pip install jupyterlab \
    && conda run -n graphban pip cache purge \
    && conda init bash \
    && echo "conda activate graphban" >> ~/.bashrc

# Copy the rest of the application code
COPY . /app/

# Expose the port of JupyterLab
EXPOSE 8889

# Set entrypoint to run JupyterLab
ENTRYPOINT ["/opt/conda/envs/protgps/bin/jupyter", "lab", "--ip='0.0.0.0'", "--port=8889", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
