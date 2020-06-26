CUDA_VISIBLE_DEVICES=0 python training.py \
    ./mimicdata/mimic3/train_full.csv \
    ./mimicdata/mimic3/vocab.csv \
    full \
    bert-tiny-caml \
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
    # 50 \
    # --pos \
    # ./mimicdata/mimic3/train_50.csv \
    # --from_scratch \
    # --redefined_tokenizer \
    # conv_attn \
