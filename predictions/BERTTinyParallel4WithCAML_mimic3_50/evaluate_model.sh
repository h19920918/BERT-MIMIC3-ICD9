python \
    ../../learn/training.py \
    ../../mimicdata/mimic3/train_50.csv \
    ../../mimicdata/mimic3/vocab.csv \
    50 \
    conv_attn \
    200 \
    --filter-size 10 \
    --num-filter-maps 50 \
    --dropout 0.1 \
    --patience 10 \
    --criterion prec_at_8 \
    --lr 5e-5 \
    --public-model \
    --test-model model.pth \
    --gpu \
    --max_sequence_length 3500 \
    --cuda_device_no 1 \
    --last_module caml_attn \
