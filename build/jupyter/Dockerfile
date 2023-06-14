ARG CORE_IMAGE

FROM ${CORE_IMAGE}

WORKDIR /tmp

# Jupyter terminal with zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
ENV SHELL=/usr/bin/zsh
RUN chsh root -s $SHELL

# Therminal dark theme
RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/terminal-extension \
    && echo '{"theme": "dark"}' > ~/.jupyter/lab/user-settings/@jupyterlab/terminal-extension/plugin.jupyterlab-settings

# Execution time & max number of outputs
RUN mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension \
    && echo '{"recordTiming": true, "maxNumberOutputs": 10000}' > ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

COPY ./build/jupyter/requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

# Clean
RUN rm -rf /tmp/*

WORKDIR /notebooks

COPY ./build/jupyter/entrypoint.sh /usr/bin/entrypoint.sh
ENTRYPOINT sh /usr/bin/entrypoint.sh