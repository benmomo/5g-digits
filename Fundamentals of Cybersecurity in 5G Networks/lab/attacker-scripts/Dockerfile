FROM debian:stable-slim

RUN apt-get update && \
    apt-get install -y curl hping3 iputils-ping && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /scripts

COPY api_bypass.sh dos_test.sh ./

RUN chmod +x api_bypass.sh dos_test.sh

ENTRYPOINT ["/bin/bash"]
