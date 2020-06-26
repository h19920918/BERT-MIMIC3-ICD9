python training.py \
    ./mimicdata/mimic3/train_50.csv \
    ./mimicdata/mimic3/vocab.csv \
    50 \
    bert-tiny-parallel4-caml \
    50 \
    --batch-size 2 \
    --filter-size 10 \
    --num-filter-maps 50 \
    --dropout 0.1 \
    --patience 10 \
    --criterion prec_at_8 \
    --lr 5e-5 \
    --embed-size 100 \
    --embed-file ./mimicdata/mimic3/processed_full.embed \
    --gpu \
    --max_sequence_length 3500 \
    --cuda_device_no 0 \
    --last_module caml_attn \
