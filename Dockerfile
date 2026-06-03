FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"
RUN mv /root/.local/bin/uv /usr/local/bin/uv

RUN mkdir -p /opt/venv
RUN mkdir -p /workspace
WORKDIR /workspace
COPY requirements.txt .

RUN groupadd -g ${HOST_GID} user && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -o -s /bin/bash user

RUN mkdir -p /home/user/.local/share \
    && chown -R ${HOST_UID}:${HOST_GID} /home/user

RUN chown -R ${HOST_UID}:${HOST_GID} /workspace
RUN chown -R ${HOST_UID}:${HOST_GID} /opt/venv

USER user

RUN uv python install 3.10.13
RUN uv venv /opt/venv --python=3.10.13

RUN echo "export PATH=/opt/venv/bin:\$PATH" >> /home/user/.bashrc
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install \
    --no-cache \
    -r requirements.txt \
    --index-strategy unsafe-best-match

CMD [ "/bin/bash" ]
