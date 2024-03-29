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
  && pip install -U --quiet langchain_experimental langchain langchain-openai langchain-community \
  && pip install chromadb \
  && pip install openai \
  && pip install tiktoken

RUN echo "==> Install jq.." \
  && pip install jq

RUN echo "==> Install streamlit.." \
  && pip install streamlit --upgrade

# Install tshark
RUN apt-get update && apt-get install -y tshark

RUN echo "==> Adding pyshark ..." \
  && pip install pyshark

RUN echo "==> Adding umap ..." \
  && pip install umap-learn

RUN echo "==> Adding pandas ..." \
  && pip install pandas

RUN echo "==> Adding scikit-learn ..." \
  && pip install scikit-learn

RUN echo "==> Adding networkx ..." \
  && pip install networkx

RUN echo "==> Adding pyvis ..." \
  && pip install pyvis

RUN echo "==> Adding dotenv ..." \
  && pip install python-dotenv

RUN echo "==> Adding Fast Embeddings ..." \
  && pip install --upgrade --quiet  fastembed

COPY /packet_raptor /packet_raptor/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]