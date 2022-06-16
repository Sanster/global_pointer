# Transformers GlobalPointer

- 苏剑林博客：
    - [Global Pointer](https://kexue.fm/archives/8373)
    - [Efficient GlobalPointer](https://spaces.ac.cn/archives/8877)
- 原版 keras 实现：https://github.com/bojone/GlobalPointer/blob/main/CLUENER_GlobalPointer.py
- CLUENER 官方测试集提交：https://www.cluebenchmarks.com/

CLUENER 结果对比

| 方法名称                                                                 | 验证集F1  |测试集F1| Head 部分参数量 |
|----------------------------------------------------------------------|--------| ---- |------------|
| CRF(from [Global Pointer](https://kexue.fm/archives/8373))           | 79.51% | 78.70% |
| GlobalPointer(from [Global Pointer](https://kexue.fm/archives/8373)) | 80.03% | 79.44% |
| Efficient GlobalPointer (from [Efficient GlobalPointer](https://spaces.ac.cn/archives/8877)) | 80.66% | 80.04% |
| GlobalPointer| 80.26% | | 102661376 |
|Efficient GlobalPointer||

训练脚本

```bash
python3 main.py \
  --model_name_or_path bert-base-chinese \
  --dataset_name ./cluener_dataset.py \
  --output_dir ./model/global_pointer2 \
  --save_total_limit 1 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --weight_decay 0.05 \
  --num_train_epochs 10 \
  --dataloader_num_workers 8 \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 100 \
  --fp16 \
  --do_train \
  --do_eval
```

评估脚本：

```bash
python3 main.py \
  --model_name_or_path ./model/global_pointer \
  --output_dir ./model/global_pointer \
  --dataset_name ./cluener_dataset.py \
  --fp16 \
  --do_eval
```

测试脚本，测试结果保存为 json:

```bash
python3 predict.py ./model/global_pointer.py gp_test.json
```