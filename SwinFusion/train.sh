python -m torch.distributed.launch --nproc_per_node=2 --master_port=1235 main_train.py --opt options/swinir/train_swinfusion_vif.json  --dist True




