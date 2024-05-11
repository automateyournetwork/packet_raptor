FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
  && pip install --break-system-packages -U --quiet langchain_experimental langchain langchain-community \
  && pip install --break-system-packages chromadb \
  && pip install --break-system-packages tiktoken

RUN echo "==> Install jq.." \
  && pip install --break-system-packages jq

RUN echo "==> Install streamlit.." \
  && pip install --break-system-packages streamlit --upgrade

# Install tshark
RUN apt-get update && apt-get install -y tshark

RUN echo "==> Adding pyshark ..." \
  && pip install --break-system-packages pyshark

RUN echo "==> Adding pandas ..." \
  && pip install --break-system-packages pandas

RUN echo "==> Adding scikit-learn ..." \
  && pip install --break-system-packages scikit-learn

RUN echo "==> Adding networkx ..." \
  && pip install --break-system-packages networkx

RUN echo "==> Adding pyvis ..." \
  && pip install --break-system-packages pyvis

RUN echo "==> Adding InstructorEmbedding ..." \
  && pip install --break-system-packages -U sentence-transformers==2.2.2 \
  && pip install --break-system-packages InstructorEmbedding

COPY /packet_raptor /packet_raptor/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]