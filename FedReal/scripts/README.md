## 一键管线

执行 `./train.sh` 即可完成以下步骤：

1. 启动 server + clients，运行 FedEXT 联邦训练；
2. 训练完成后，自动根据保存的客户端模型生成特征/标签数据集；
3. 使用生成的特征训练尾部分类器。

所有超参数都在 `train.sh` 开头集中配置，包括联邦阶段、特征导出以及尾部训练。
若需开启 WandB 日志，将 `USE_WANDB` 设为 `1` 并配置 `WANDB_PROJECT` 等参数；脚本会自动为服务器、所有客户端以及尾部分类器创建对应的 run，并在 `Global/step` 上对齐轮次。

---

如需手动控制，可直接调用 `launch.py`：

```bash
python scripts/launch.py \
  --num_clients 20 \
  --bind 0.0.0.0:11432 \
  --server_addr 127.0.0.1:11432 \
  --data_root ./dataset \
  --dataset_name Cifar10 \
  --num_classes 10 \
  --rounds 50 \
  --local_epochs 5 \
  --batch_size 64 \
  --lr 0.01 \
  --momentum 0.9 \
  --sample_fraction 1.0 \
  --seed 42 \
  --model_name resnet18 \
  --feature_dim 512 \
  --encoder_ratio 0.7 \
  --max_message_mb 128 \
  --server_warmup_sec 2 \
  --stagger_sec 0.2 \
  --env_omp1 \
  --gpus 0 \
  --log_dir logs \
  --run_dir ./runs/Cifar10/FedEXT/manual_run
```
