# ð¤ GlobalPointer

- èåæåå®¢ï¼
    - [Global Pointer](https://kexue.fm/archives/8373)
    - [Efficient GlobalPointer](https://spaces.ac.cn/archives/8877)
- åç keras å®ç°ï¼https://github.com/bojone/GlobalPointer/blob/main/CLUENER_GlobalPointer.py
- CLUENER å®æ¹æµè¯éæäº¤ï¼https://www.cluebenchmarks.com/

CLUENER ç»æå¯¹æ¯

| æ¹æ³åç§°                                                                                         | éªè¯éF1  |æµè¯éF1| åæ°é |
|----------------------------------------------------------------------------------------------|--------| ---- |------------|
| CRF(from [Global Pointer](https://kexue.fm/archives/8373))                                   | 79.51% | 78.70% |
| GlobalPointer(from [Global Pointer](https://kexue.fm/archives/8373))                         | 80.03% | 79.44% |
| Efficient GlobalPointer (from [Efficient GlobalPointer](https://spaces.ac.cn/archives/8877)) | 80.66% | 80.04% |
| GlobalPointer(w/ RoPE)                                                                       | 80.26% | | 102661376 |
| GlobalPointer(w/o RoPE)                                                                      | 79.3%  | | 102661376 |
| Efficient GlobalPointer(w/ RoPE)                                                             | 80.64% || 101790868 |
| Efficient GlobalPointer(w/o RoPE)                                                            | 79.57% || 101778068 |

è®­ç»èæ¬ï¼

- éè¿ `--global_pointer_head` åæ¢ `GlobalPointer` å `EfficientGlobalPointer`
- éè¿ `--rope` åæ¢è¦ä¸è¦å æè½¬ä½ç½®ç¼ç  `RoPE`

```bash
python3 main.py \
  --model_name_or_path bert-base-chinese \
  --dataset_name ./cluener_dataset.py \
  --output_dir ./model/efficient_global_pointer_no_rope \
  --save_total_limit 1 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --global_pointer_head EfficientGlobalPointer \
  --weight_decay 0.05 \
  --num_train_epochs 10 \
  --dataloader_num_workers 8 \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 100 \
  --rope True \
  --fp16 \
  --do_train \
  --do_eval
```

å¯¹éªè¯éè¿è¡è¯ä¼°ï¼

```bash
python3 main.py \
  --model_name_or_path ./model/global_pointer \
  --output_dir ./model/global_pointer \
  --dataset_name ./cluener_dataset.py \
  --fp16 \
  --do_eval
```

è·æµè¯èæ¬ï¼æµè¯ç»æä¿å­ä¸º json:

```bash
python3 predict.py test ./model/global_pointer.py gp_test.json
```

ç´æ¥è¾å¥ input çé¢æµç»æ

```bash
python3 predict.py predict ./model/global_pointer.py
```
