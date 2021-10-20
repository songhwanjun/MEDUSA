import os
import torch

PARALLEL_WORLD = ["127.0.0.1"]
PARALLEL_PORTS = [6946]
MY_RANK = 0
os.environ["NCCL_DEBUG"] = "info"

n = torch.cuda.device_count()
print(n)

world_size = max(len(PARALLEL_WORLD), 1)

# This is 8 GPUs setting with 32 batch size
batch_size = '4'
image_height = 300
image_width = 500
num_workers = '2'
epochs = '150'
lr_drop = 100
print_freq = 200
path = '/home/Research/MyData/COCO2017'

conf = '--batch_size %s ' \
       '--num_workers %s '\
       '--epochs %s ' \
       '--lr_drop %s ' \
       '--print_freq %s ' \
       '--path %s ' % (batch_size, num_workers, epochs, lr_drop,
                            print_freq, path)

command = 'python -m torch.distributed.launch ' \
          '--nproc_per_node=%d ' \
          '--nnodes=%d ' \
          '--node_rank=%d ' \
          '--master_addr=%s ' \
          '--master_port=%d ' \
          '--use_env main.py %s' % (n, world_size, MY_RANK, PARALLEL_WORLD[0], PARALLEL_PORTS[0], conf)

print(command)
os.system(command)