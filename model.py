#model_marketplace.config
# {"token_length": "4018", "accuracy": "70", "precision": "fp16", "sampling_frequency:": "44100", "mono": true, "fps": "74", "resolution": "480", "image_width": "1080", "image_height": "1920", "framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name": "DeepSeek-V3",
#       "value": "deepseek-ai/DeepSeek-V3",
#       "size": 20,
#       "paramasters": "685B",
#       "tflops": 14, 
#       "vram": 20,
#       "nodes": 10
#     },
# {
#       "name": "DeepSeek-V3-bf16",
#       "value": "opensourcerelease/DeepSeek-V3-bf16",
#       "size": 1500,
#       "paramasters": "684B",
#       "tflops": 80, 
#       "vram": 48,
#       "nodes": 10
#     }
#   ], "cuda": "11.4", "task":["text-generation", "text-classification", "text-summarization", "text-ner", "question-answering"]}
import os
from typing import List, Dict, Optional
import uuid
# from label_studio_ml.model import LabelStudioMLBase
# from label_studio_ml.response import ModelResponse
# from transformers import pipeline
import torch
# https://gitee.com/hf-models/seamless-m4t-v2-large?skip_mobile=true
# from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

# processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
# model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# # from text
# text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
# audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()

# # from audio
# audio, orig_freq =  torchaudio.load("https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav")
# audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
# audio_inputs = processor(audios=audio, return_tensors="pt")
# audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()

# MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/opt-125m')
# _model = pipeline('text-generation', model=MODEL_NAME)


# class HuggingFaceLLM(LabelStudioMLBase):
#     """Custom ML Backend model
#     """

#     MAX_LENGTH = int(os.getenv('MAX_LENGTH', 50))

#     def setup(self):
#         """Configure any paramaters of your model here
#         """
#         self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

#     def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
#         """ Write your inference logic here
#             :param tasks: [AIxBlock tasks in JSON format](https://labelstud.io/guide/task_format.html)
#             :param context: [AIxBlock context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
#             :return model_response
#                 ModelResponse(predictions=predictions) with
#                 predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
#         """
#         from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Text')
#         predictions = []
#         for task in tasks:
#             text = self.preload_task_data(task, task['data'][value])
#             result = _model(text, max_length=self.MAX_LENGTH)
#             generated_text = result[0]['generated_text']
#             # cut the `text` prefix
#             generated_text = generated_text[len(text):].strip()
#             predictions.append({
#                 'result': [{
#                     'from_name': from_name,
#                     'to_name': to_name,
#                     'type': 'textarea',
#                     'value': {
#                         'text': [generated_text]
#                     }
#                 }],
#                 'model_version': self.get('model_version')
#             })
        
#         return ModelResponse(predictions=predictions, model_version=self.get("model_version"))

from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
import torch.distributed as dist
import torch
import subprocess
import json
import subprocess
import threading
import requests
# from dashboard import promethus_grafana
from loguru import logger
import numpy as np
from function_ml import connect_project, download_dataset, upload_checkpoint
from logging_class import start_queue, write_log
from prompt import qa_without_context
import time
from mcp.server.fastmcp import FastMCP
import zipfile
from huggingface_hub import (
    HfFolder, 
    login,
    whoami,
    ModelCard,
    upload_file,
    create_repo
)

hf_token = os.getenv("HF_TOKEN", "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")
HfFolder.save_token(hf_token)


hf_access_token = "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"
login(token=hf_access_token)
CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


# audio sample rate 44100 vs 48000
AUDIO_SAMPLE_RATE = 44100 # 16000.0
MAX_INPUT_AUDIO_LENGTH = 1800  # in seconds
DEFAULT_TARGET_LANGUAGE = "vie"
# Language dict
language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}


# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "afr",
    "amh",
    "arb",
    "ary",
    "arz",
    "asm",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cym",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "eus",
    "fin",
    "fra",
    "gaz",
    "gle",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khk",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lug",
    "luo",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mni",
    "mya",
    "nld",
    "nno",
    "nob",
    "npi",
    "nya",
    "ory",
    "pan",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
    "yor",
    "yue",
    "zsm",
    "zul",
]

# Target langs:
# S2ST / T2ST
s2st_target_language_codes = [
    "eng",
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cym",
    "dan",
    "deu",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
]

def read_dataset(file_path):
    # Kiểm tra xem thư mục /content/ có tồn tại không
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        # Kiểm tra xem có file json nào không
        for file in files:
            if file.endswith(".json"):
            # Đọc file json
                with open(os.path.join(file_path, file), "r") as f:
                    data = json.load(f)

                return data
    return None

def is_correct_format(data_json):
    try:
        for item in data_json:
            if not all(key in item for key in ['instruction', 'input', 'output']):
                return False
        return True
    except Exception as e:
        return False
    
def conver_to_hf_dataset(data_json):
    formatted_data = []
    for item in data_json:
        for annotation in item['annotations']:
            question = None
            answer = None
            for result in annotation['result']:
                if result['from_name'] == 'question':
                    question = result['value']['text'][0]
                elif result['from_name'] == 'answer':
                    answer = result['value']['text'][0]
            if question and answer:
                formatted_data.append({
                    'instruction': item['data']['text'],
                    'input': question,
                    'output': answer
                })
    return formatted_data

    # dataset = Dataset.from_list(formatted_data)
class MyModel(AIxBlockMLBase):
    @mcp.tool()
    def action(self, command, **kwargs):
        logger.info(f"Received command: {command} with args: {kwargs}")
        if command.lower() == "train":
            try:
                model_id = kwargs.get("model_id", "meta-llama/Llama-3.2-1B-Instruct")
                dataset_id = kwargs.get(
                    "dataset_id", "autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus"
                )

                push_to_hub = kwargs.get("push_to_hub", True)
                hf_model_id = kwargs.get(
                    "hf_model_id", "meta-llama/Llama-3.2-1B-Instruct"
                )
                push_to_hub_token = kwargs.get(
                    "push_to_hub_token", "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"
                )
                framework = kwargs.get("framework", "huggingface")
                task = kwargs.get("task", "text-generation")
                prompt = kwargs.get("prompt", "")
                trainingArguments = kwargs.get("TrainingArguments", None)
                cuda_debug = kwargs.get("cuda_debug", False)

                json_file = "training_args.json"
                absolute_path = os.path.abspath(json_file)

                with open(absolute_path, "w") as f:
                    json.dump(trainingArguments, f)
                logger.info(f"Training arguments: {trainingArguments}")

                if cuda_debug == True:
                    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
                    os.environ["NCCL_DEBUG"] = "INFO"

                os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
                os.environ["TORCH_USE_CUDA_DSA"] = "0"
                clone_dir = os.path.join(os.getcwd())
                project_id = kwargs.get("project_id", 0)
                token = kwargs.get("token", "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")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", 1)
                rank = kwargs.get("rank", 0)
                master_add = kwargs.get("master_add", "127.0.0.1")
                master_port = kwargs.get("master_port", "23456")
                host_name = kwargs.get("host_name", HOST_NAME)
                instruction_field = kwargs.get("prompt_field", "prompt")
                input_field = kwargs.get("input_field", "task_description")
                output_field = kwargs.get("output_field", "response")
                log_queue, logging_thread = start_queue(channel_log)
                epoch = kwargs.get("epoch", 1)
                batch_size = kwargs.get("batch_size", 1)
                write_log(log_queue)
                channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"
                username = ""
                hf_model_name = ""

                try:
                    user = whoami(token=push_to_hub_token)['name']
                    hf_model_name = f"{user}/{hf_model_id}"
                except Exception as e:
                    hf_model_name = "Token not correct"
                    print(e)
                    
                CHANNEL_STATUS[channel_name] = {
                    "status": "training",
                    "hf_model_id": hf_model_name,
                    "command": command,
                    "created_at": time.time(),
                }
                print(f"🚀 Đã bắt đầu training kênh: {channel_name}")

                def func_train_model(
                    clone_dir,
                    project_id,
                    token,
                    checkpoint_version,
                    checkpoint_id,
                    dataset_version,
                    dataset_id,
                    model_id,
                    world_size,
                    rank,
                    master_add,
                    master_port,
                    prompt,
                    json_file,
                    channel_log,
                    hf_model_id,
                    push_to_hub,
                    push_to_hub_token,
                    host_name,
                    epoch,
                    batch_size
                ):

                    dataset_path = None
                    project = connect_project(host_name, token, project_id)

                    if dataset_version and dataset_id and project:
                        dataset_path = os.path.join(
                            clone_dir, f"datasets/{dataset_version}"
                        )

                        if not os.path.exists(dataset_path):
                            data_path = os.path.join(clone_dir, "data_zip")
                            os.makedirs(data_path, exist_ok=True)

                            dataset_name = download_dataset(project, dataset_id, data_path)
                            print(dataset_name)
                            if dataset_name:
                                data_zip_dir = os.path.join(data_path, dataset_name)

                                with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                    zip_ref.extractall(dataset_path)

                                extracted_files = os.listdir(dataset_path)
                                zip_files = [
                                    f for f in extracted_files if f.endswith(".zip")
                                ]

                                if len(zip_files) == 1:
                                    inner_zip_path = os.path.join(
                                        dataset_path, zip_files[0]
                                    )
                                    print(
                                        f"🔁 Found inner zip file: {inner_zip_path}, extracting..."
                                    )
                                    with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
                                        inner_zip.extractall(dataset_path)
                                    os.remove(inner_zip_path)

                        train_dir = os.path.join(dataset_path, "train/train_manifest.json")
                        validation_dir = os.path.join(dataset_path, "validation/validation_manifest.json")
                    else:
                        base_dir = "fleurs"
                        subprocess.run(["mkdir", "-p", f"{base_dir}/train"], check=True)

                        # Chạy lệnh m4t_prepare_dataset
                        subprocess.run([
                            "venv/bin/m4t_prepare_dataset",
                            "--name", "google/fleurs",
                            "--source_lang", "eng",
                            "--target_lang", "eng",
                            "--split", "train",
                            "--save_dir", f"{base_dir}/train"
                        ], check=True)

                        subprocess.run(["mkdir", "-p", f"{base_dir}/validation"], check=True)

                        # Chạy lệnh m4t_prepare_dataset
                        subprocess.run([
                            "venv/bin/m4t_prepare_dataset",
                            "--name", "google/fleurs",
                            "--source_lang", "eng",
                            "--target_lang", "eng",
                            "--split", "validation",
                            "--save_dir", f"{base_dir}/validation"
                        ], check=True)
                        
                        train_dir = os.path.join(base_dir, "train/train_manifest.json")
                        validation_dir = os.path.join(base_dir, "validation/validation_manifest.json")

                    make_dir = os.path.join(os.getcwd(), "checkpoints")
                    os.makedirs(make_dir, exist_ok=True)
                    subprocess.run(
                        ("whereis accelerate"),
                        shell=True,
                    )
                    print("===Train===")

                    if int(world_size) > 1:
                        if rank == 0:
                            subprocess.run([
                                "venv/bin/torchrun",
                                "--rdzv-backend=c10d",
                                "--rdzv-endpoint=localhost:0",
                                "--nnodes", str(world_size),
                                "--nproc-per-node", str(world_size * torch.cuda.device_count()),
                                "--node_rank", str(rank),
                                "--master_addr", "127.0.0.1",
                                "--master_port", "23456",
                                "--no-python",  
                                "m4t_finetune",
                                "--mode", "SPEECH_TO_TEXT",
                                "--train_dataset", train_dir,
                                "--eval_dataset", validation_dir,
                                "--learning_rate", "1e-6",
                                "--batch_size", batch_size,
                                "--warmup_steps", "100",
                                "--max_epochs", epoch,
                                "--patience", "5",
                                "--model_name", "seamlessM4T_medium",
                                "--save_model_to", f"{make_dir}/checkpoint.pt"
                            ], check=True)
                        else:
                            subprocess.run([
                                "venv/bin/torchrun",
                                "--rdzv-backend=c10d",
                                "--rdzv-endpoint=localhost:0",
                                "--nnodes", str(world_size),
                                "--nproc-per-node", str(world_size * torch.cuda.device_count()),
                                "--node_rank", str(rank),
                                "--master_addr", master_add,
                                "--master_port", master_port,
                                "--no-python",  
                                "m4t_finetune",
                                "--mode", "SPEECH_TO_TEXT",
                                "--train_dataset", train_dir,
                                "--eval_dataset", validation_dir,
                                "--learning_rate", "1e-6",
                                "--batch_size", batch_size,
                                "--warmup_steps", "100",
                                "--max_epochs", epoch,
                                "--patience", "5",
                                "--model_name", "seamlessM4T_medium",
                                "--save_model_to", f"{make_dir}/checkpoint.pt"
                            ], check=True)
                    else:
                        subprocess.run([
                            "venv/bin/m4t_finetune",
                            "--train_dataset", train_dir,
                            "--eval_dataset", validation_dir,
                            "--batch_size", str(batch_size),
                            "--eval_steps", "1000",
                            "--learning_rate", "0.00005",
                            "--patience", "10",
                            "--max_epochs", str(epoch),
                            "--model_name", "seamlessM4T_medium",
                            "--save_model_to", f"{make_dir}/checkpoint.pt"
                        ], check=True)

                    checkpoint_path = os.path.join(make_dir, "checkpoint.pt")

                    user = whoami(token=push_to_hub_token)['name']
                    repo_id = f"{user}/{hf_model_id}"

                    # Nếu repo chưa tồn tại, tạo mới
                    create_repo(repo_id=repo_id, token=push_to_hub_token, exist_ok=True)

                    # Tạo ModelCard metadata (nếu chưa có)
                    try:
                        card = ModelCard.load(repo_id, token=push_to_hub_token)
                    except Exception:
                        card = ModelCard("")

                    if not card.text.lstrip().startswith("---"):
                        yaml_metadata = (
                            "---\n"
                            "license: apache-2.0\n"
                            "language: en\n"
                            "tags:\n"
                            "  - speech\n"
                            "  - translation\n"
                            f"model_name: {hf_model_id}\n"
                            "---\n\n"
                        )
                        card.text = yaml_metadata + card.text

                    # Bổ sung phần Citations
                    sections = card.text.split("## ")
                    new_sections = []
                    for section in sections:
                        if section.lower().startswith("citations"):
                            new_section = (
                                "Citations\n\n"
                                "This model was fine-tuned using custom data and training scripts.\n\n"
                                "© 2025 YourTeamName. All rights reserved.\n"
                            )
                            new_sections.append(new_section)
                        else:
                            new_sections.append(section)
                    card.text = "## ".join(new_sections)

                    # Save ModelCard locally
                    readme_path = "README.md"
                    with open(readme_path, "w") as f:
                        f.write(card.text)

                    # Upload README.md
                    upload_file(
                        path_or_fileobj=readme_path,
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        token=push_to_hub_token,
                        commit_message="Upload README"
                    )
                    fallback_path = os.path.join(os.getcwd(), "tokenizer.model")

                    # Kiểm tra trong checkpoints/
                    checkpoints_path = os.path.join(os.getcwd(), "checkpoints", "tokenizer.model")

                    # Nếu file tồn tại ở checkpoints thì dùng
                    if os.path.exists(checkpoints_path):
                        tokenizer_model_path = checkpoints_path
                    else:
                        tokenizer_model_path = fallback_path

                    print("✅ Dùng tokenizer tại:", tokenizer_model_path)

                    upload_file(
                        path_or_fileobj=tokenizer_model_path,
                        path_in_repo="tokenizer.model",
                        repo_id=repo_id,
                        token=push_to_hub_token,
                        commit_message="Upload tokenizer"
                    )

                    # ✅ Upload checkpoint (đặt tên chuẩn theo Transformers nếu có thể)
                    upload_file(
                        path_or_fileobj=checkpoint_path,
                        path_in_repo="seamlessM4T_medium.pt",
                        repo_id=repo_id,
                        token=push_to_hub_token,
                        commit_message="Upload fine-tuned checkpoint"
                    )

                    print("✅ Đã upload README.md và checkpoint lên Hugging Face Hub!")
                    

                    CHANNEL_STATUS[channel_name]["status"] = "done"
                    output_dir = "./data/checkpoint"
                    print(push_to_hub)
                    if push_to_hub:
                        import datetime

                        output_dir = "./data/checkpoint"
                        now = datetime.datetime.now()
                        date_str = now.strftime("%Y%m%d")
                        time_str = now.strftime("%H%M%S")
                        version = f"{date_str}-{time_str}"

                        upload_checkpoint(project, version, output_dir)

                train_thread = threading.Thread(
                    target=func_train_model,
                    args=(
                        clone_dir,
                        project_id,
                        token,
                        checkpoint_version,
                        checkpoint_id,
                        dataset_version,
                        dataset_id,
                        model_id,
                        world_size,
                        rank,
                        master_add,
                        master_port,
                        prompt,
                        absolute_path,
                        channel_log,
                        hf_model_id,
                        push_to_hub,
                        push_to_hub_token,
                        host_name,
                        epoch,
                        batch_size
                    ),
                )
                train_thread.start()

                return {
                    "message": "train completed successfully",
                    "channel_name": channel_name,
                }
            except Exception as e:
                return {"message": f"train failed: {e}"}

        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir ./logs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        
        elif command.lower() == "dashboard":
            link = promethus_grafana.generate_link_public("ml_00")
            return {"Share_url": link}
          
        elif command.lower() == "predict":
                # import torch
                from seamless_communication.inference import Translator
                local_model_path = '/app/models-hf/facebook/seamless-m4t-v2-large'
                translator = Translator(
                    model_name_or_card="seamlessM4T_v2_large",
                    vocoder_name_or_card="vocoder_v2",
                    device=device,
                    dtype=dtype,
                    apply_mintox=True,
                )
            # try:
                data = kwargs.get("audio",None)
                if not data: 
                    data = kwargs.get("data", "")
                    
                source_language = kwargs.get("source","en")
                target_language = kwargs.get("target","en")
                prompt = kwargs.get("prompt", "")
                model_id = kwargs.get("model_id", "")

                text = kwargs.get("source", None)
                if not text:
                    text = kwargs.get("text", None)

                token_length = kwargs.get("token_lenght", "")
                task = kwargs.get("task", "")
                voice = kwargs.get("voice", "")
                max_gen_len = kwargs.get("max_gen_len", "")
                temperature = kwargs.get("temperature", "")
                top_p = kwargs.get("top_p", "")
                seed = kwargs.get("seed", "")   
                project_id = kwargs.get("project_id")
                token = kwargs.get("token")
                def decode_base64_to_audio(base64_audio, output_file="output.wav"):
                        # Giải mã Base64 thành nhị phân
                        import base64
                        # import os  
                        file_path = os.path.join(os.path.dirname(__file__), output_file)
                        audio_data = base64.b64decode(base64_audio)
                        
                        # Ghi dữ liệu nhị phân vào file âm thanh
                        with open(file_path, "wb") as audio_file:
                            audio_file.write(audio_data)
                        return file_path
                # source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
                # target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                def download_audio(audio_url, save_path):
                    # Tạo request để tải video từ URL
                    response = requests.get(audio_url, stream=True)
                    
                    # Kiểm tra nếu request thành công
                    if response.status_code == 200:
                        with open(save_path, 'wb') as audio_file:
                            for chunk in response.iter_content(chunk_size=1024):
                                if chunk:
                                    audio_file.write(chunk)
                        print(f"audio has been downloaded and saved to {save_path}")
                        return save_path  # Trả về đường dẫn đến video đã tải về
                    else:
                        print(f"Failed to download audio. Status code: {response.status_code}")
                        return None

                # Thay audio_url bằng URL audio thật của bạn
                audio_url = data
                if "http://" in audio_url or "https://" in audio_url:
                    input_audio= download_audio(data,"audio.wav")
                else:
                    input_audio= decode_base64_to_audio(base64_audio=data)
                def preprocess_audio(input_audio: str) -> None:
                    # import torchaudio
                    arr, org_sr = torchaudio.load(input_audio)
                    
                    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
                    # max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
                    # if new_arr.shape[1] > max_length:
                    #     new_arr = new_arr[:, :max_length]
                        # gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.") int(AUDIO_SAMPLE_RATE)
                    torchaudio.save(input_audio, new_arr, sample_rate=AUDIO_SAMPLE_RATE)

                # S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])
                # TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])
                # LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}
                # https://raw.githubusercontent.com/facebookresearch/seamless_communication/main/docs/m4t/README.md
                # def preprocess_audio(input_audio: str) -> None:
                #     import torchaudio
                #     arr, org_sr = torchaudio.load(input_audio)
                    
                #     new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=org_sr)
                #     # max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
                #     # if new_arr.shape[1] > max_length:
                #     #     new_arr = new_arr[:, :max_length]
                #         # gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.") int(AUDIO_SAMPLE_RATE)
                #     torchaudio.save(input_audio, new_arr, sample_rate=org_sr)

                # # S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])
                # # TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])
                LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}
                # # https://raw.githubusercontent.com/facebookresearch/seamless_communication/main/docs/m4t/README.md
                def run_s2st(
                    input_audio: str, source_language: str, target_language: str
                ) -> tuple[tuple[int, np.ndarray] | None, str]:
                    # import torchaudio
                    preprocess_audio(input_audio)
                    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    out_texts, out_audios = translator.predict(
                        input=input_audio,
                        task_str="S2ST",
                        src_lang=source_language_code,
                        tgt_lang=target_language_code,
                    )
                    out_text = str(out_texts[0])
                   
                    unique_filename = str(uuid.uuid4())  # Generate a UUID and convert it to a string
                    file_extension = ".wav"  # Replace with the desired file extension
                    file_path = os.path.join("./", unique_filename + file_extension)
                    import scipy
                    audio_array = out_audios.audio_wavs[0].cpu().detach().numpy().squeeze()
                    audio_array /=1.414
                    audio_array *= 32767
                    audio_array = audio_array.astype(np.int16)
                    scipy.io.wavfile.write(file_path, rate=out_audios.sample_rate, data=audio_array)
                    # self.upload_raw_file(file_path, project_id, token)
                    # # Save the translated audio generation.
                    # torchaudio.save(
                    #     file_path,
                    #      out_audios.audio_wavs[0][0].cpu(),
                    #     sample_rate=out_audios.sample_rate
                    # )
                    print(f"file_path:{file_path} out_text:{out_text}")
                    return file_path, out_text #(int(AUDIO_SAMPLE_RATE), out_wav)


                def run_s2tt(input_audio: str, source_language: str, target_language: str) -> str:
                    preprocess_audio(input_audio)
                    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    out_texts, _ = translator.predict(
                        input=input_audio,
                        task_str="S2TT",
                        src_lang=source_language_code,
                        tgt_lang=target_language_code,
                    )
                    return str(out_texts[0])


                def run_t2st(input_text: str, source_language: str, target_language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
                    # import torchaudio
                    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    
                    out_texts, out_audios = translator.predict(
                        input=input_text,
                        task_str="T2ST",
                        src_lang=source_language_code,
                        tgt_lang=target_language_code
                    )
                    out_text = str(out_texts[0])
                    # out_wav = out_audios.audio_wavs[0].cpu().detach().numpy()
                    # return (int(AUDIO_SAMPLE_RATE), out_wav), out_text
              
                    unique_filename = str(uuid.uuid4())  # Generate a UUID and convert it to a string
                    file_extension = ".wav"  # Replace with the desired file extension
                    file_path = os.path.join("./", unique_filename + file_extension)
                    import scipy
                    audio_array = out_audios.audio_wavs[0].cpu().detach().numpy().squeeze()
                    audio_array /=1.414
                    audio_array *= 32767
                    audio_array = audio_array.astype(np.int16)
                    scipy.io.wavfile.write(file_path, rate=out_audios.sample_rate, data=audio_array)
                    # self.upload_raw_file(file_path, project_id, token)
                    # # Save the translated audio generation.
                    # torchaudio.save(
                    #     file_path,
                    #      out_audios.audio_wavs[0][0].cpu(),
                    #     sample_rate=out_audios.sample_rate
                    # )
                    return file_path, out_text #(int(AUDIO_SAMPLE_RATE), out_wav)


                def run_t2tt(input_text: str, source_language: str, target_language: str) -> str:
                    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    out_texts, _ =  translator.predict(
                        input=input_text,
                        task_str="T2TT",
                        src_lang=source_language_code,
                        tgt_lang=target_language_code,
                    )
                    return str(out_texts[0])


                def run_asr(input_audio: str, target_language: str) -> str:
                    preprocess_audio(input_audio)
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    out_texts, _ =  translator.predict(
                        input=input_audio,
                        task_str="ASR",
                        src_lang=target_language_code,
                        tgt_lang=target_language_code,
                    )
                    return str(out_texts[0])
# {
#   "command": "predict",
#   "params": {
#     "source":"[ngôn ngử nguồn]",
#     "target":"[ngôn ngử đích]",
#     "prompt": "",
#     "model_id": "",<= thay đổi ở đây FE hardcode vào
#     "token_lenght": 50,
#     "task": "translation",
#     "text": "[nội dung texxt nếu có]",
#     "max_gen_len":1024, 
#     "temperature":0.9, 
#     "top_p":0.5, 
#     "seed":0
#   },
#   "project": "215"
# } 
                
                if len(voice)>0:
                    
                    audio_file = decode_base64_to_audio(voice["data"])
                    file_path = "unity_on_device.ptl"

                    if not os.path.exists(file_path):
                        url = "https://huggingface.co/facebook/seamless-m4t-unity-small/resolve/main/unity_on_device.ptl"
                        response = requests.get(url)

                        # Lưu file
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                    # import torchaudio
                    audio_input, _ = torchaudio.load(audio_file) # Load waveform using torchaudio

                    s2st_model = torch.jit.load(file_path)

                    with torch.no_grad():
                        prompt, units, waveform = s2st_model(audio_input, tgt_lang="eng")

                predictions = []
                base64_output = []
                generated_url=""
                generated_text=""
# speech-to-speech-translation (S2ST)
# speech-to-text-translation (S2TT)
# text-to-speech-translation (T2ST)
# text-to-text-translation (T2TT)
# automatic-speech-recognition (ASR).
                # input_audio= video_path #decode_base64_to_audio(base64_audio=data,output_file="input.wav")
                if task == "speech-to-speech-translation":
                    audio_path,result_text = run_s2st(input_audio=input_audio,source_language=source_language,target_language=target_language)
                    import base64
                    from io import BytesIO
                    with open(audio_path, "rb") as fh:
                        buffer = BytesIO(fh.read())
                    buffer.seek(0)
                    base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    generated_text = result_text
                    generated_url = f"/downloads?path={audio_path}"
                elif task == "speech-to-text-translation":
                    result = run_s2tt(input_audio=input_audio, source_language=source_language,target_language=target_language)
                    generated_text = result

                elif task == "text-to-speech-translation":
                    audio_path,result_text = run_t2st ( input_text=text,source_language=source_language,target_language=target_language)
                    import base64
                    from io import BytesIO
                    with open(audio_path, "rb") as fh:
                        buffer = BytesIO(fh.read())
                    buffer.seek(0)
                    base64_output = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    generated_text = result_text
                    generated_url = f"/downloads?path={audio_path}"
                elif task == "text-to-text-translation":
                   
                    result = run_t2tt ( input_text=text,source_language=source_language,target_language=target_language)
                    generated_text = result

                elif task == "automatic-speech-recognition":
                    result = run_asr(input_audio=input_audio,target_language=target_language)
                    generated_text = result
                elif task == "automatic-speech-recognition-segment":
                    import math
                    from simuleval.data.segments import SpeechSegment, EmptySegment
                    # from seamless_communication.streaming.agents.seamless_streaming_s2st import (
                    #     SeamlessStreamingS2STVADAgent,
                    # )

                    from simuleval.utils.arguments import cli_argument_list
                    from simuleval import options


                    from typing import Union, List
                    from simuleval.data.segments import Segment, TextSegment
                    from simuleval.agents.pipeline import TreeAgentPipeline
                    from simuleval.agents.states import AgentStates

                    import io
                    # import json
                    import matplotlib as mpl
                    import matplotlib.pyplot as plt
                    # import mmap
                    import soundfile
                    # import torchaudio
                    # import torch

                    from collections import defaultdict
                    from IPython.display import Audio, display
                    # from pathlib import Path
                    from pydub import AudioSegment

                    # from seamless_communication.inference import Translator
                    # from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover

                    SAMPLE_RATE = 44100


                    class AudioFrontEnd:
                        def __init__(self, wav_file, segment_size) -> None:
                            self.samples, self.sample_rate = soundfile.read(wav_file)
                            # self.sample_rate = SAMPLE_RATE
                            # print(len(self.samples), self.samples[:100])
                            # self.samples = self.samples  # .tolist()
                            self.segment_size = segment_size
                            self.step = 0

                        def send_segment(self):
                            """
                            This is the front-end logic in simuleval instance.py
                            """

                            num_samples = math.ceil(self.segment_size / 1000 * self.sample_rate)

                            if self.step < len(self.samples):
                                if self.step + num_samples >= len(self.samples):
                                    samples = self.samples[self.step :]
                                    is_finished = True
                                else:
                                    samples = self.samples[self.step : self.step + num_samples]
                                    is_finished = False
                                self.step = min(self.step + num_samples, len(self.samples))

                                segment = SpeechSegment(
                                    content=samples,
                                    sample_rate=self.sample_rate,
                                    finished=is_finished,
                                )
                            else:
                                # Finish reading this audio
                                segment = EmptySegment(
                                    finished=True,
                                )
                            return segment


                    class OutputSegments:
                        def __init__(self, segments: Union[List[Segment], Segment]):
                            if isinstance(segments, Segment):
                                segments = [segments]
                            self.segments: List[Segment] = [s for s in segments]

                        @property
                        def is_empty(self):
                            return all(segment.is_empty for segment in self.segments)

                        @property
                        def finished(self):
                            return all(segment.finished for segment in self.segments)


                    def get_audiosegment(samples, sr):
                        b = io.BytesIO()
                        soundfile.write(b, samples, samplerate=sr, format="wav")
                        b.seek(0)
                        return AudioSegment.from_file(b)


                    def reset_states(system, states):
                        if isinstance(system, TreeAgentPipeline):
                            states_iter = states.values()
                        else:
                            states_iter = states
                        for state in states_iter:
                            state.reset()


                    def get_states_root(system, states) -> AgentStates:
                        if isinstance(system, TreeAgentPipeline):
                            # self.states is a dict
                            return states[system.source_module]
                        else:
                            # self.states is a list
                            return system.states[0]


                    def plot_s2st(source_file, target_samples, target_fs, intervals, delays, prediction_lists):
                        mpl.rcParams["axes.spines.left"] = False
                        mpl.rcParams["axes.spines.right"] = False
                        mpl.rcParams["axes.spines.top"] = False
                        mpl.rcParams["axes.spines.bottom"] = False

                        source_samples, source_fs = soundfile.read(source_file)

                        _, axes = plt.subplots(5, sharex=True, figsize=(25, 5))
                        for ax in axes:
                            ax.set_yticks([])

                        axes[0].plot(
                            np.linspace(0, len(source_samples) / source_fs, len(source_samples)),
                            source_samples,
                        )

                        axes[1].plot(
                            np.linspace(0, len(target_samples) / target_fs, len(target_samples)),
                            target_samples,
                        )

                        start = 0
                        for seg_index in range(len(intervals)):
                            start, duration = intervals[seg_index]
                            offset = delays["s2st"][seg_index]

                            samples = target_samples[
                                int((start) / 1000 * target_fs) : int(
                                    (start + duration) / 1000 * target_fs
                                )
                            ]

                            # Uncomment this if you want to see the segments without speech playback delay
                            axes[2].plot(
                                offset / 1000 + np.linspace(0, len(samples) / target_fs, len(samples)),
                                -seg_index * 0.05 + np.array(samples),
                            )
                            axes[4].plot(
                                start / 1000 + np.linspace(0, len(samples) / target_fs, len(samples)),
                                np.array(samples),
                            )

                        from pydub import AudioSegment
                        print("Output translation (without input)")
                        # display(Audio(target_samples, rate=target_fs))
                        print("Output translation (overlay with input)")
                        source_seg = get_audiosegment(source_samples, source_fs) + AudioSegment.silent(duration=3000)
                        target_seg = get_audiosegment(target_samples, target_fs)
                        output_seg = source_seg.overlay(target_seg)
                        # display(output_seg)

                        delay_token = defaultdict(list)
                        d = delays["s2tt"][0]
                        for token, delay in zip(prediction_lists["s2tt"], delays["s2tt"]):
                            if delay != d:
                                d = delay
                            delay_token[d].append(token)
                        for key, value in delay_token.items():
                            axes[3].text(
                                key / 1000, 0.2, " ".join(value), rotation=40
                            )

                    def build_streaming_system(model_configs, agent_class):
                        parser = options.general_parser()
                        parser.add_argument("-f", "--f", help="a dummy argument to fool ipython", default="1")

                        agent_class.add_args(parser)
                        args, _ = parser.parse_known_args(cli_argument_list(model_configs))
                        system = agent_class.from_args(args)
                        return system


                    def run_streaming_inference(system, audio_frontend, system_states, tgt_lang):
                        # NOTE: Here for visualization, we calculate delays offset from audio
                        # *BEFORE* VAD segmentation.
                        # In contrast for SimulEval evaluation, we assume audios are pre-segmented,
                        # and Average Lagging, End Offset metrics are based on those pre-segmented audios.
                        # Thus, delays here are *NOT* comparable to SimulEval per-segment delays
                        delays = {"s2st": [], "s2tt": []}
                        prediction_lists = {"s2st": [], "s2tt": []}
                        speech_durations = []
                        curr_delay = 0
                        target_sample_rate = None

                        while True:
                            input_segment = audio_frontend.send_segment()
                            input_segment.tgt_lang = tgt_lang
                            curr_delay += len(input_segment.content) / SAMPLE_RATE * 1000
                            if input_segment.finished:
                                # a hack, we expect a real stream to end with silence
                                get_states_root(system, system_states).source_finished = True
                            # Translation happens here
                            output_segments = OutputSegments(system.pushpop(input_segment, system_states))
                            if not output_segments.is_empty:
                                for segment in output_segments.segments:
                                    # NOTE: another difference from SimulEval evaluation -
                                    # delays are accumulated per-token
                                    if isinstance(segment, SpeechSegment):
                                        pred_duration = 1000 * len(segment.content) / segment.sample_rate
                                        speech_durations.append(pred_duration)
                                        delays["s2st"].append(curr_delay)
                                        prediction_lists["s2st"].append(segment.content)
                                        target_sample_rate = segment.sample_rate
                                    elif isinstance(segment, TextSegment):
                                        delays["s2tt"].append(curr_delay)
                                        prediction_lists["s2tt"].append(segment.content)
                                        print(curr_delay, segment.content)
                            if output_segments.finished:
                                print("End of VAD segment")
                                reset_states(system, system_states)
                            if input_segment.finished:
                                # an assumption of SimulEval agents -
                                # once source_finished=True, generate until output translation is finished
                                # assert output_segments.finished
                                break
                        return delays, prediction_lists, speech_durations, target_sample_rate


                    def get_s2st_delayed_targets(delays, target_sample_rate, prediction_lists, speech_durations):
                        # get calculate intervals + durations for s2st
                        intervals = []

                        start = prev_end = prediction_offset = delays["s2st"][0]
                        target_samples = [0.0] * int(target_sample_rate * prediction_offset / 1000)

                        for i, delay in enumerate(delays["s2st"]):
                            start = max(prev_end, delay)

                            if start > prev_end:
                                # Wait source speech, add discontinuity with silence
                                target_samples += [0.0] * int(
                                    target_sample_rate * (start - prev_end) / 1000
                                )

                            target_samples += prediction_lists["s2st"][i]
                            duration = speech_durations[i]
                            prev_end = start + duration
                            intervals.append([start, duration])
                        return target_samples, intervals
                    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
                        SeamlessStreamingS2STJointVADAgent,
                    )


                    print("building system from dir")
                    agent_class = SeamlessStreamingS2STJointVADAgent
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    tgt_lang = target_language_code
                    source_segment_size = 320  # milliseconds
                    model_configs = dict(
                        source_segment_size=source_segment_size,
                        device=device,
                        dtype="fp16",
                        min_starting_wait_w2vbert=192,
                        decision_threshold=0.5,
                        min_unit_chunk_size=50,
                        no_early_stop=True,
                        max_len_a=0,
                        max_len_b=100,
                        task="s2st",
                        tgt_lang=tgt_lang,
                        block_ngrams=True,
                        detokenize_only=True,
                    )
                    system = build_streaming_system(model_configs, agent_class)
                    print("finished building system")
                    
                    audio_frontend = AudioFrontEnd(
                        wav_file=input_audio,
                        segment_size=source_segment_size,
                    )

                    system_states = system.build_states()

                    # you can pass tgt_lang at inference time to change the output lang.
                    # SeamlessStreaming supports 36 speech output languages, see https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md#supported-languages
                    # in the Target column for `Sp` outputs.
                    delays, prediction_lists, speech_durations, target_sample_rate = run_streaming_inference(
                        system, audio_frontend, system_states, tgt_lang
                    )
                    # result = run_asr(input_audio=input_audio,target_language=target_language)
                    generated_text = prediction_lists
                    base64_output = speech_durations
                     
# {
#     'result': [{
#         'from_name': "generated_text",
#         'to_name': "text_output", 
#         'type': 'textarea',
#         'value': [
#             {
#                 "id": segment_id,
#                 "meta": {
#                     "text": [
#                         f"{file_name},{format_name},{bit_rate},{_channel},{duration},{folder_name}" 
#                     ]
#                 },
#                 "type": "textarea",
#                 "value": {
#                     "end":  segment_end,
#                     "text": [
#                         "text"
#                     ],
#                     "start": segment_start
#                 },
#                 "origin": "manual",
#                 "to_name": "audio",
#                 "from_name": "transcription",
#                 "original_length": duration
#             }
#         ]
#     }],
#     'model_version': ""
# }
                else:
                    raise ValueError(f"Task type '{task}' not supported")
                
                predictions.append({
                    'result': [{
                        'from_name': "generated_text",
                        'to_name': "text_output", #audio
                        'type': 'textarea',
                        'value': {
                            'data': base64_output,
                            "url": generated_url, 
                            'text': generated_text
                        }
                    }],
                    'model_version': ""
                })
                print(predictions)
                return {"message": "predict completed successfully", "result": predictions}
            # except Exception as e:
            #     print(e)
            #     return {"message": "predict failed", "result": None}
        
        elif command.lower() == "prompt_sample":
                task = kwargs.get("task", "")
                if task == "question-answering":
                    prompt_text = f"""
                   Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question using only a single word or phrase from the context without repeating the question or adding any extra explanation: 
                    {{question}}

                    Answer:
                    """
                elif task == "text-classification":
                   prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                
                elif task == "summarization":
                    prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                return {"message": "prompt_sample completed successfully", "result":prompt_text}
        
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "llama_recipes/finetuning.py"])
            return {"message": "Done", "result": "Done"}
        
        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}
        
        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}
        
    @mcp.tool()
    def model(self, **kwargs):
        
        import gradio as gr 
        task = kwargs.get("task", "text-to-speech-translation")

        # def load_model():
        from seamless_communication.inference import Translator
        translator = Translator(
                model_name_or_card="seamlessM4T_v2_large",
                vocoder_name_or_card="vocoder_v2",
                device=device,
                dtype=dtype,
                apply_mintox=True,
            )
            
        LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}
        S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])
        TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])
        ASR_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
        S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
        T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
        T2ST_TARGET_LANGUAGE_NAMES = S2ST_TARGET_LANGUAGE_NAMES
       
        # def preprocess_audio(input_audio: str) -> None:
        #         import torchaudio
        #         arr, org_sr = torchaudio.load(input_audio)
                
        #         new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=org_sr)
        #         # max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        #         # if new_arr.shape[1] > max_length:
        #         #     new_arr = new_arr[:, :max_length]
        #             # gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.") int(AUDIO_SAMPLE_RATE)
        #         torchaudio.save(input_audio, new_arr, sample_rate=org_sr)

        #         # S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])
        #         # TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])
        #         LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}
        # def preprocess_audio(input_audio: str) -> None:
        #     import torchaudio
        #     arr, org_sr = torchaudio.load(input_audio)
      
        #     new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=org_sr)
        #     torchaudio.save(input_audio, new_arr, sample_rate=org_sr)

        #         # https://raw.githubusercontent.com/facebookresearch/seamless_communication/main/docs/m4t/README.md
        
        # def run_s2st(
        #     input_audio: str, source_language: str, target_language: str
        # ) -> tuple[tuple[int, np.ndarray] | None, str]:
        #     # translator = load_model()
           

        #     source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
        #     target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
        #     out_texts, out_audios = translator.predict(
        #         input=input_audio,
        #         task_str="S2ST",
        #         src_lang=source_language_code,
        #         tgt_lang=target_language_code,
        #     )
        #     out_text = str(out_texts[0])
        #     import uuid      
        #     import os         
        #     import numpy as np
        #     unique_filename = str(uuid.uuid4())  # Generate a UUID and convert it to a string
        #     file_extension = ".wav"  # Replace with the desired file extension
        #     file_path = os.path.join("./", unique_filename + file_extension)
        #     print("start saving audio")
           
        #     import scipy
        #     # if torch.cuda.is_available():
        #     #     audio_array = out_audios.audio_wavs[0].detach().numpy().argmax()
        #     # else:
        #     audio_array = out_audios.audio_wavs[0].cpu().detach().numpy().squeeze()
        #     audio_array /=1.414
        #     audio_array *= 32767
        #     # if torch.cuda.is_available():
        #     #     audio_array = audio_array
        #     # else:
        #     audio_array = audio_array.astype(np.int16)
          
        #     scipy.io.wavfile.write(file_path, rate=out_audios.sample_rate, data=audio_array)
        #     print("finish saving audio")
        #     print(f"file_path:{file_path} out_text:{out_text}")
        #     return file_path, out_text
        def preprocess_audio(input_audio: str) -> None:
                    # import torchaudio
                    arr, org_sr = torchaudio.load(input_audio)
                    
                    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
                    # max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
                    # if new_arr.shape[1] > max_length:
                    #     new_arr = new_arr[:, :max_length]
                        # gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.") int(AUDIO_SAMPLE_RATE)
                    torchaudio.save(input_audio, new_arr, sample_rate=AUDIO_SAMPLE_RATE)

                # S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])
                # TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])
                # LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}
                # https://raw.githubusercontent.com/facebookresearch/seamless_communication/main/docs/m4t/README.md
        def run_s2st(
                    input_audio: str, source_language: str, target_language: str
                ) -> tuple[tuple[int, np.ndarray] | None, str]:
                    # import torchaudio
                    preprocess_audio(input_audio)
                    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
                    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
                    out_texts, out_audios = translator.predict(
                        input=input_audio,
                        task_str="S2ST",
                        src_lang=source_language_code,
                        tgt_lang=target_language_code,
                    )
                    out_text = str(out_texts[0])
                   
                    unique_filename = str(uuid.uuid4())  # Generate a UUID and convert it to a string
                    file_extension = ".wav"  # Replace with the desired file extension
                    file_path = os.path.join("./", unique_filename + file_extension)
                    import scipy
                    audio_array = out_audios.audio_wavs[0].cpu().detach().numpy().squeeze()
                    audio_array /=1.414
                    audio_array *= 32767
                    audio_array = audio_array.astype(np.int16)
                    scipy.io.wavfile.write(file_path, rate=out_audios.sample_rate, data=audio_array)
                    # self.upload_raw_file(file_path, project_id, token)
                    # # Save the translated audio generation.
                    # torchaudio.save(
                    #     file_path,
                    #      out_audios.audio_wavs[0][0].cpu(),
                    #     sample_rate=out_audios.sample_rate
                    # )
                    print(f"file_path:{file_path} out_text:{out_text}")
                    return file_path, out_text #(int(AUDIO_SAMPLE_RATE), out_wav)
        
        def run_s2tt(input_audio: str,input_audio_mic: str, source_language: str, target_language: str) -> str:
            if input_audio_mic != None:
                #  preprocess_audio(input_audio_mic)
                 input_audio=input_audio_mic
            preprocess_audio(input_audio)
            source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
            target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
            out_texts, _ = translator.predict(
                input=input_audio,
                task_str="S2TT",
                src_lang=source_language_code,
                tgt_lang=target_language_code,
            )
            return str(out_texts[0])

        def run_t2st(input_text: str, source_language: str, target_language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
            # translator = load_model()

            # import torchaudio
            source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
            target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
            
            out_texts, out_audios = translator.predict(
                input=input_text,
                task_str="T2ST",
                src_lang=source_language_code,
                tgt_lang=target_language_code
            )
            out_text = str(out_texts[0])
            # out_wav = out_audios.audio_wavs[0].cpu().detach().numpy()
            # return (int(AUDIO_SAMPLE_RATE), out_wav), out_text
        
            unique_filename = str(uuid.uuid4())  # Generate a UUID and convert it to a string
            file_extension = ".wav"  # Replace with the desired file extension
            file_path = os.path.join("./", unique_filename + file_extension)
            import scipy
            audio_array = out_audios.audio_wavs[0].cpu().detach().numpy().squeeze()
            audio_array /=1.414
            audio_array *= 32767
            audio_array = audio_array.astype(np.int16)
            scipy.io.wavfile.write(file_path, rate=out_audios.sample_rate, data=audio_array)
            # self.upload_raw_file(file_path, project_id, token)
            # # Save the translated audio generation.
            # torchaudio.save(
            #     file_path,
            #      out_audios.audio_wavs[0][0].cpu(),
            #     sample_rate=out_audios.sample_rate
            # )
            return file_path, out_text #(int(AUDIO_SAMPLE_RATE), out_wav)

        def run_t2tt(input_text: str, source_language: str, target_language: str) -> str:
            # translator = load_model()

            source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
            target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
            out_texts, _ = translator.predict(
                input=input_text,
                task_str="T2TT",
                src_lang=source_language_code,
                tgt_lang=target_language_code,
            )
            return str(out_texts[0])


        def run_asr(input_audio: str,input_audio_file: str, target_language: str) -> str:
            preprocess_audio(input_audio)
            # translator = load_model()
            target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
            out_texts, _ = translator.predict(
                input=input_audio,
                task_str="ASR",
                src_lang=target_language_code,
                tgt_lang=target_language_code,
            )
            return str(out_texts[0])

        with gr.Blocks() as demo_s2st:
            with gr.Row():
                with gr.Column():
                    with gr.Row() as audio_box:
                        input_audio = gr.Audio(label="Input speech", type="filepath")
                        # input_audio_mic = gr.Audio(
                        #     label="Input speech",
                        #     type="filepath",
                        #     source="microphone",
                        # )
                        # input_audio_file = gr.Audio(
                        #     label="Input speech",
                        #     type="filepath",
                        #     source="upload",
                        # )
                    with gr.Group():
                        # input_microphone =gr.Audio(sources=["microphone"], type="filepath",label="microphone", streaming=True)
                        # input_audio =gr.Audio(sources=["upload"], type="filepath",label="upload")
                        source_language = gr.Dropdown(
                            label="Source language",
                            choices=ASR_TARGET_LANGUAGE_NAMES,
                            value="English",
                        )
                        target_language = gr.Dropdown(
                            label="Target language",
                            choices=S2ST_TARGET_LANGUAGE_NAMES,
                            value="vie",
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    with gr.Group():
                        output_audio = gr.Audio(
                            label="Translated speech",
                            autoplay=False,
                            streaming=False,
                            type="numpy",
                        )
                        output_text = gr.Textbox(label="Translated text")

            # gr.Examples(
            #     examples=[
            #         ["/app/sample_input.mp3", "English", "French"],
            #         ["/app/sample_input.mp3", "English", "Mandarin Chinese"],
            #         ["/app/sample_input.mp3", "English", "Vietnamese"],
            #         ["/app/sample_input.mp3", "English", "Spanish"],
            #     ],
            #     inputs=[input_audio, source_language, target_language],
            #     outputs=[output_audio, output_text],
            #     fn=run_s2st,
            #     api_name=False,
            # )

            btn.click(
                fn=run_s2st,
                inputs=[input_audio, source_language, target_language],
                outputs=[output_audio, output_text],
                api_name="s2st",
            )

        with gr.Blocks() as demo_s2tt:
            with gr.Row():
                with gr.Row() as audio_box:
                    input_audio_mic = gr.Audio(
                        label="Input speech",
                        type="filepath",
                        sources="microphone",
                    )
                    input_audio_file = gr.Audio(
                        label="Input speech",
                        type="filepath",
                        sources="upload",
                    )
                with gr.Column():
                    with gr.Group():
                        # input_audio = gr.Audio(label="Input speech", type="filepath")
                        source_language = gr.Dropdown(
                            label="Source language",
                            choices=ASR_TARGET_LANGUAGE_NAMES,
                            value="English",
                        )
                        target_language = gr.Dropdown(
                            label="Target language",
                            choices=S2TT_TARGET_LANGUAGE_NAMES,
                            value=DEFAULT_TARGET_LANGUAGE,
                        )
                    btn = gr.Button("Translate")
                with gr.Column():
                    output_text = gr.Textbox(label="Translated text")

            # gr.Examples(
            #     examples=[
            #         ["/app/sample_input.mp3", "English", "French"],
            #         ["/app/sample_input.mp3", "English", "Mandarin Chinese"],
            #         ["/app/sample_input.mp3", "English", "Vietnamese"],
            #         ["/app/sample_input.mp3", "English", "Spanish"],
            #     ],
            #     inputs=[input_audio_mic,input_audio_file, source_language, target_language],
            #     outputs=output_text,
            #     fn=run_s2tt,
            #     api_name=False,
            # )

            btn.click(
                fn=run_s2tt,
                inputs=[input_audio_mic,input_audio_file, source_language, target_language],
                outputs=output_text,
                api_name="s2tt",
            )

        with gr.Blocks() as demo_t2st:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        with gr.Row():
                            source_language = gr.Dropdown(
                                label="Source language",
                                choices=TEXT_SOURCE_LANGUAGE_NAMES,
                                value="English",
                            )
                            target_language = gr.Dropdown(
                                label="Target language",
                                choices=T2ST_TARGET_LANGUAGE_NAMES,
                                value=DEFAULT_TARGET_LANGUAGE,
                            )
                    btn = gr.Button("Translate")

                with gr.Column():
                    with gr.Group():
                        output_audio = gr.Audio(
                            label="Translated speech",
                            autoplay=False,
                            streaming=False,
                            type="numpy",
                        )
                        output_text = gr.Textbox(label="Translated text")

            # gr.Examples(
            #     examples=[
            #         [
            #             "My favorite animal is the elephant.",
            #             "English",
            #             "French",
            #         ],
            #         [
            #             "My favorite animal is the elephant.",
            #             "English",
            #             "Vietnamee",
            #         ],
            #         [
            #             "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
            #             "English",
            #             "Mandarin Chinese",
            #         ],
            #         [
            #             "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
            #             "English",
            #             "Spanish",
            #         ],
            #     ],
            #     inputs=[input_text, source_language, target_language],
            #     outputs=[output_audio, output_text],
            #     fn=run_t2st,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=run_t2st,
                inputs=[input_text, source_language, target_language],
                outputs=[output_audio, output_text],
                api_name="t2st",
            )

        with gr.Blocks() as demo_t2tt:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        with gr.Row():
                            source_language = gr.Dropdown(
                                label="Source language",
                                choices=TEXT_SOURCE_LANGUAGE_NAMES,
                                value="English",
                            )
                            target_language = gr.Dropdown(
                                label="Target language",
                                choices=T2TT_TARGET_LANGUAGE_NAMES,
                                value=DEFAULT_TARGET_LANGUAGE,
                            )
                    btn = gr.Button("Translate")
                with gr.Column():
                    output_text = gr.Textbox(label="Response text")

            # gr.Examples(
            #     examples=[
            #         [
            #             "My favorite animal is the elephant.",
            #             "English",
            #             "French",
            #         ],
            #         [
            #             "My favorite animal is the elephant.",
            #             "English",
            #             "Mandarin Chinese",
            #         ],
            #         [
            #             "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
            #             "English",
            #             "Vietnamee",
            #         ],
            #         [
            #             "Meta AI's Seamless M4T model is democratising spoken communication across language barriers",
            #             "English",
            #             "Spanish",
            #         ],
            #     ],
            #     inputs=[input_text, source_language, target_language],
            #     outputs=output_text,
            #     fn=run_t2tt,
            #     api_name=False,
            # )

            gr.on(
                triggers=[input_text.submit, btn.click],
                fn=run_t2tt,
                inputs=[input_text, source_language, target_language],
                outputs=output_text,
                api_name="t2tt",
            )

        with gr.Blocks() as demo_asr:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        #  with gr.Row() as audio_box:
                            input_audio_mic = gr.Audio(
                                label="Input speech",
                                type="filepath",
                                sources="microphone",
                            )
                            input_audio_file = gr.Audio(
                                label="Input speech",
                                type="filepath",
                                sources="upload",
                            )
                        # input_audio = gr.Audio(label="Input speech", type="filepath")
                            target_language = gr.Dropdown(
                                label="Target language",
                                choices=ASR_TARGET_LANGUAGE_NAMES,
                                value=DEFAULT_TARGET_LANGUAGE,
                            )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response text")

            # gr.Examples(
            #     examples=[
            #          ["/app/sample_input.mp3", "English"],
            #         ["/app/sample_input.mp3", "English"],
            #         ["/app/sample_input.mp3", "English"],
            #         ["/app/sample_input.mp3", "English"],
            #     ],
            #     inputs=[input_audio, target_language],
            #     outputs=output_text,
            #     fn=run_asr,
            #     api_name=False,
            # )

            btn.click(
                fn=run_asr,
                inputs=[input_audio_file,input_audio_mic, target_language],
                outputs=output_text,
                api_name="asr",
            )

        DESCRIPTION = """\
        # SeamlessM4T v2

        [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
        translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.
        This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
        translation and more, without relying on multiple separate models.
        """

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)

            with gr.Tabs():
                # if task == "speech-to-speech-translation":
                    with gr.Tab(label="S2ST"):
                        demo_s2st.render()
                # elif task == "speech-to-text-translation":
                    with gr.Tab(label="S2TT"):
                        demo_s2tt.render()
                # elif task == "text-to-speech-translation":
                    with gr.Tab(label="T2ST"):
                        demo_t2st.render()
                # elif task == "text-to-text-translation":
                    with gr.Tab(label="T2TT"):
                        demo_t2tt.render()
                # elif task == "automatic-speech-recognition-segment":
                    with gr.Tab(label="ASR"):
                        demo_asr.render()
                # elif task == "automatic-speech-recognition":
                    # with gr.Tab(label="ASR"):
                    #     demo_asr.render()
                # elif task == "translation":
                    # with gr.Tab(label="T2TT"):
                    #     demo_t2tt.render()
                # else:
                #     return {"share_url": "", 'local_url': ""}
                        

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    @mcp.tool()
    def model_trial(self, project, **kwargs):
        import gradio as gr 
        return {"message": "Done", "result": "Done"}


        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

           
            def predict(input_img):
            
                # result = self.action(project, "predict",collection="",data={"img":input_img})
                # print(result)
                # if result['result']:
                #     boxes = result['result']['boxes']
                #     names = result['result']['names']
                #     labels = result['result']['labels']
                    
                #     for box, label in zip(boxes, labels):
                #         box = [int(i) for i in box]
                #         label = int(label)
                #         input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                #         # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                #         input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                return input_img
            
            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'
                
            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(project, "train",collection="",data=dataset_choosen)
                return result['message']

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                # import os
                checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
                            project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)
                
                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)
            
            def upload_file(file):
                return "File uploaded!"
            
            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):   
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])
                    
                    gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False), 
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False             
                    )


                # with gr.TabItem("Webcam", id=1):    
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):    
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):  
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Row():
                                #get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")

                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]
                                
                                dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True, type="value")
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML("""
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>""")
                                
                                dataset_choosen.select(download_btn, None, download_link)
                                
                                #when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown("## Upload your sample dataset to have a trial training")
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(predict, gr.File(elem_classes=["upload_image"],file_types=['tar','zip']), 
                                gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False             
                    )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(trial_training, dataset_choosen, None)
                
                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)
   
        return {"share_url": share_url, 'local_url': local_url}
    
    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import send_from_directory,request
        file_path = request.args.get('path')
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)