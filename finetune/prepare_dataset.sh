#!/bin/bash
# 'google/fleurs', 'speechcolab/gigaspeech'
# notice: modify load tsv meta files in seamless_communication/datasets/huggingface.py
# https://huggingface.co/datasets/google/fleurs
# https://huggingface.co/datasets/speechcolab/gigaspeech
# {'id': 91,
#  'num_samples': 385920,
#  'path': '/home/patrick/.cache/huggingface/datasets/downloads/extracted/310a663d52322700b3d3473cbc5af429bd92a23f9bc683594e70bc31232db39e/home/vaxelrod/FLEURS/oss2_obfuscated/af_za/audio/train/17797742076841560615.wav',
#  'audio': {'path': '/home/patrick/.cache/huggingface/datasets/downloads/extracted/310a663d52322700b3d3473cbc5af429bd92a23f9bc683594e70bc31232db39e/home/vaxelrod/FLEURS/oss2_obfuscated/af_za/audio/train/17797742076841560615.wav',
#   'array': array([ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,
#          -1.1205673e-04, -8.4638596e-05, -1.2731552e-04], dtype=float32),
#   'sampling_rate': 16000},
#  'raw_transcription': 'Dit is nog nie huidiglik bekend watter aantygings gemaak sal word of wat owerhede na die seun gelei het nie maar jeugmisdaad-verrigtinge het in die federale hof begin',
#  'transcription': 'dit is nog nie huidiglik bekend watter aantygings gemaak sal word of wat owerhede na die seun gelei het nie maar jeugmisdaad-verrigtinge het in die federale hof begin',
#  'gender': 0,
#  'lang_id': 0,
#  'language': 'Afrikaans',
#  'lang_group_id': 3}

# Using the cached tokenizer of seamlessM4T_v2_large. Set `force` to `True` to download again.
# 2024-12-10 08:07:43,639 INFO -- finetune.569: Finetune Params: FinetuneParams(model_name='seamlessM4T_v2_large', save_model_path=PosixPath('/app/data/m4t-traindata/checkpoint.pt'), finetune_mode=<FinetuneMode.TEXT_TO_SPEECH: 'TEXT_TO_SPEECH'>, float_dtype=torch.float16, max_epochs=10, label_smoothing=0.2, warmup_steps=100, log_steps=10, eval_steps=50, patience=5, learning_rate=1e-06, train_batch_size=5, eval_batch_size=5, device=device(type='cuda'))
# Using the cached checkpoint of seamlessM4T_v2_large. Set `force` to `True` to download again.
# Using the cached tokenizer of seamlessM4T_v2_large. Set `force` to `True` to download again.
# Using the cached tokenizer of seamlessM4T_v2_large. Set `force` to `True` to download again.
# 2024-12-10 08:07:57,626 INFO -- seamless_communication.cli.m4t.finetune.trainer.569: Freeze s2t: True, freeze t2u: False
# /usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:28: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn("The verbose parameter is deprecated. Please use get_last_lr() "
# 2024-12-10 08:07:57,634 INFO -- seamless_communication.cli.m4t.finetune.trainer.569: Start Finetuning
# 2024-12-10 08:07:57,635 INFO -- seamless_communication.cli.m4t.finetune.trainer.569: Evaluation Step 0...
# Traceback (most recent call last):
#   File "/usr/local/lib/python3.10/dist-packages/seamless_communication/cli/m4t/finetune/trainer.py", line 131, in forward
#     raise NotImplementedError(
# NotImplementedError: T2U finetuning implemented only for UnitYT2UModel

mkdir -p /app/data/m4t-traindata
m4t_prepare_dataset --name google/fleurs --source_lang eng --target_lang vie --split train --save_dir /app/data/m4t-traindata
m4t_prepare_dataset --name google/fleurs --source_lang eng --target_lang vie --split validation --save_dir /app/data/m4t-traindata

python3.10 /app/finetune/truncate-fleurs-corpus.py /app/data/m4t-traindata/train_manifest.json 600 #2030 
python3.10 /app/finetune/truncate-fleurs-corpus.py /app/data/m4t-traindata/validation_manifest.json 600 #2030 

DATASET_DIR=/app/data/m4t-traindata

# torchrun \
#    --rdzv-backend=c10d \
#    --rdzv-endpoint=localhost:0 \
#    --nnodes=1 \
#    --nproc-per-node=1  \
#    --no-python \
#   m4t_finetune \
#    --mode SPEECH_TO_TEXT \
#    --train_dataset $DATASET_DIR/train_manifest.json  \
#    --eval_dataset $DATASET_DIR/validation_manifest.json \
#    --learning_rate 1e-6 \
#    --warmup_steps 100 \
#    --max_epochs 30 \
#    --patience 5 \
#    --model_name seamlessM4T_v2_large \
#    --save_model_to $DATASET_DIR/checkpoint.pt

m4t_finetune \
   --mode SPEECH_TO_TEXT \
   --train_dataset $DATASET_DIR/train_manifest.json  \
   --eval_dataset $DATASET_DIR/validation_manifest.json \
   --learning_rate 1e-6 \
   --warmup_steps 100 \
   --max_epochs 10 \
   --patience 5 \
   --learning_rate 1e-6 \
   --batch_size 3 \
   --model_name seamlessM4T_v2_large \
   --save_model_to $DATASET_DIR/checkpoint.pt
