# FROM python:3.10-slim

# ENV PYTHONUNBUFFERED=True \
#     PORT=9090

# WORKDIR /app
# COPY requirements.txt .
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libgl1 \
#         libglib2.0-0
# RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# COPY . ./
# EXPOSE 6006
# EXPOSE 9090
# CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app

FROM wowai/base-hf:v1.12.0
WORKDIR /app
COPY . ./
# WORKDIR /tmp
COPY requirements.txt .

ENV MODEL_DIR=/data/models
ENV RQ_QUEUE_NAME=default
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV PORT=9090
ENV AIXBLOCK_USE_REDIS=false
ENV HOST_NAME=https://dev-us-west-1.aixblock.io

RUN apt install -y ffmpeg

# COPY uwsgi.ini /etc/uwsgi
RUN apt-get -qq update && \
   DEBIAN_FRONTEND=noninteractive \ 
   apt-get install --no-install-recommends --assume-yes \
    git
RUN apt install libpq-dev -y uwsgi
RUN apt install build-essential
RUN apt install -y libpq-dev python3-dev
RUN python3.10 -m pip install psycopg2
RUN python3.10 -m pip install python-box
RUN apt install -y nvidia-cuda-toolkit --fix-missing

RUN python3.10 -m pip install --upgrade colorama
RUN python3.10 -m pip install aixblock-sdk
RUN apt-get update
RUN apt install -y nvidia-cuda-toolkit --fix-missing
RUN python3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu12.2

RUN apt-get update
RUN apt-get -qq -y install curl --fix-missing

RUN python3.10 -m pip install -U "huggingface_hub[cli]"

RUN --mount=type=cache,target=/root/.cache 
RUN python3.10 -m pip install -r requirements.txt

# RUN python3.10 -m pip install fairseq2
RUN python3.10 -m pip install pydub sentencepiece
RUN python3.10 -m pip install git+https://github.com/facebookresearch/seamless_communication.git

# RUN python3.10 -m pip install -q fairseq2 gradio
# RUN python3.10 -m pip install -q https://huggingface.co/spaces/facebook/seamless-m4t-v2-large/resolve/main/whl/seamless_communication-1.0.0-py3-none-any.whl
RUN apt -y install -qq aria2
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/seamless-m4t-v2-large/resolve/main/seamlessM4T_v2_large.pt -d /app/models -o seamlessM4T_v2_large.pt
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/seamless-m4t-v2-large/resolve/main/spm_char_lang38_tc.model -d /app/models -o spm_char_lang38_tc.model
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/seamless-m4t-v2-large/resolve/main/vocoder_v2.pt -d /app/models -o vocoder_v2.pt
# WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN python3.10 -m pip install --upgrade Flask
RUN python3.10 -m pip install gradio==3.50.0
RUN python3.10 -m pip install ffprobe-python
RUN python3.10 -m pip install ffmpeg-python==0.2.0
RUN python3.10 -m pip install huggingface_hub[hf_transfer]
RUN python3.10 -m pip install huggingface_hub
RUN python3.10 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
# RUN huggingface-cli download facebook/seamless-m4t-v2-large --cache-dir /app/models/  --token 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 --force
# RUN huggingface-cli download facebook/seamless-m4t-v2-large --repo-type model --repo-id facebook/seamless-m4t-v2-large --token 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 --force
COPY . ./
EXPOSE 9090 6006 12345
CMD exec gunicorn --preload --bind :${PORT} --workers 1 --threads 1 --timeout 0 _wsgi:app

