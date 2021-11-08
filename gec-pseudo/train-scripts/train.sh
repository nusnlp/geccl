cd path/to/your/fairseq/folder

gpu=$1
margin=0.25
trade_off=0.5
MODEL_DIR=$2
FAIRSEQ_DIR=fairseq
MAX_TOKENS=4096
PSEUDO_PATH=ldc_giga.spell_error.finetune.checkpoint_best.pt

CUDA_VISIBLE_DEVICES=$gpu python -u $FAIRSEQ_DIR/train.py path/to/data/bin \
    --save-dir $MODEL_DIR \
    --num-sources 6 \
    --arch transformer_vaswani_wmt_en_de_big \
    --max-tokens $MAX_TOKENS \
    --task multi_gec \
    --restore-file $PSEUDO_PATH \
    --optimizer adam \
    --lr 0.00003 \
    -s src \
    -t tgt \
    --margin $margin \
    --trade_off $trade_off \
    --dropout 0.3 \
    --clip-norm 1.0 \
    --criterion new_max_margin_loss \
    --label-smoothing 0.1 \
    --adam-betas '(0.9,0.98)' \
    --log-format simple \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --max-epoch 10 \
    --seed 1
