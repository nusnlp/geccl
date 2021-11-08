gpu=$1
input=$2
dataset=$3
MODEL_DIR=$4
OUTPUT_DIR=$5

beam=5

SUBWORD_NMT=subword-nmt
FAIRSEQ_DIR=fairseq
BPE_MODEL_DIR=bpe
PREPROCESS=vocab

cd path/to/gec-pseudo


pwd
test_src_name="$OUTPUT_DIR/test.bpe.src.$dataset"
test_nbest_name="$OUTPUT_DIR/test.nbest.tok.$dataset"
test_best_name="$OUTPUT_DIR/test.best.tok.$dataset"
$SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/bpe_code.trg.dict_bpe8000 < $input > $test_src_name

echo Generating...
CUDA_VISIBLE_DEVICES=$gpu python -u ${FAIRSEQ_DIR}/interactive.py $PREPROCESS \
    --path ${MODEL_DIR} \
    --beam ${beam} \
    --nbest ${beam} \
    --no-progress-bar \
    -s src_bpe8000 \
    -t trg_bpe8000 \
    --buffer-size 1024 \
    --batch-size 32 \
    --log-format simple \
    --remove-bpe \
    < $test_src_name > $test_nbest_name

cat $test_nbest_name | grep "^H"  | python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if (i % ${beam} == 0) ]); print(x)" | cut -f3 > $test_best_name
sed -i '$d' $test_best_name
