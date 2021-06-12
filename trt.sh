python tools/pytorch2trt.py \
    configs/condlanenet/curvelanes/curvelanes_small_test.py \
    ./curvelanes_small.pth \
    --out curvelanes_small.engine \
    --shape 320 800
