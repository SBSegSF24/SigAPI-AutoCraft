FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NOWARNINGS=yes

# Install Python and necessary dependencies
RUN apt-get update -qq && apt-get upgrade -y && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    build-essential \
    curl \
    wget \
    vim && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /sigapi
COPY . ./

RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x reproduzir_sigapi_autocraft.sh
CMD ["./reproduzir_sigapi_autocraft.sh"]
