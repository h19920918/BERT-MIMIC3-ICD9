CUDA_VISIBLE_DEVICES=3 python -m pdb training.py \
    ./mimicdata/bio-mimic3/train_50.csv \
    ./mimicdata/bio-mimic3/vocab.csv \
    50 \
    biobert \
    50 \
    --filter-size 10 \
    --num-filter-maps 50 \
    --dropout 0.2 \
    --patience \
    10 \
    --criterion prec_at_8 \
    --lr 5e-5 \
    --embed-file ./mimicdata/bio-mimic3/processed_full.embed \
    --gpu \
    --pos \
    --redefined_tokenizer \
    # conv_attn \
