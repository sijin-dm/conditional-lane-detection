CUDA_VISIBLE_DEVICES=0 python tools/condlanenet/curvelanes/vis_curvelanes.py \
    configs/condlanenet/curvelanes/curvelanes_small_test_custom.py \
    ./curvelanes_small.pth --show_dst outputs
