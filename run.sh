#python tools/condlanenet/curvelanes/vis_curvelanes.py configs/condlanenet/curvelanes/curvelanes_small_test.py ./curvelanes_small.pth --evaluate --show --show_dst outputs
#python tools/condlanenet/curvelanes/vis_curvelanes.py configs/condlanenet/curvelanes/curvelanes_small_test_custom.py ./curvelanes_small.pth --show --show_dst outputs
python tools/pytorch2trt.py \
    configs/condlanenet/curvelanes/curvelanes_small_test.py \
    ./curvelanes_small.pth \
    --out curvelanes_small.engine \
    --shape 1280 720