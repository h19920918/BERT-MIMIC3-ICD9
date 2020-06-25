CUDA_VISIBLE_DEVICES=3 python training.py \
    ./mimicdata/mimic3/train_50.csv \
    ./mimicdata/mimic3/vocab.csv \
    50 \
    bert-tiny \
    50 \
    --filter-size 10 \
    --num-filter-maps 50 \
    --dropout 0.2 \
    --patience \
    10 \
    --criterion prec_at_8 \
    --lr 5e-5 \
    --embed-file ./mimicdata/mimic3/processed_full.embed \
    --gpu \
    --batch-size 4 \
    --last_module caml_attn \
    --pretrain-batch-size 4 \
    --pretrain \
    # --pos \
    # --pretrain-batch-size 2 \
    # --redefined_tokenizer \
    # --pretrain \
    # --from_scratch \
    # conv_attn \
