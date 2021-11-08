cd path/to/your/BART-GEC/folder

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=1400
UPDATE_FREQ=4
BART_PATH=path/to/trained/BART/model

CUDA_VISIBLE_DEVICES=$1 python3 train.py path/to/data/bin \
    --save-dir $4 \
    --num-sources 5 \
    --log-format simple \
    --log-interval 100 \
    --seed 1 \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task multi_gec \
    --margin $2 \
    --trade_off $3 \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch bart_large \
    --criterion new_max_margin_loss \
    --label-smoothing 0.1 \
    --dropout 0.3 --attention-dropout 0.3 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --max-epoch 20 \
    --find-unused-parameters;
