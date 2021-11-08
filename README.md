# GEC-CL: Grammatical Error Correction with Contrastive Learning in Low Error Density Domains

> Hannan Cao, Wenmian Yang, Hwee Tou Ng. Grammatical Error Correction with Contrastive Learning in Low Error Density Domains. In Findings of EMNLP 2021 [\[paper\]](https://aclanthology.org/2021.findings-emnlp.419.pdf)[\[code\]](https://github.com/nusnlp/geccl)

Codes in the two directories are the GEC-CL systems for [GEC-PD](https://github.com/butsugiri/gec-pseudodata) and [GEC-BART](https://github.com/Katsumata420/generic-pretrained-GEC/tree/master/BART-GEC)

## Runtime Environment:

This system has been tested in the following environment.
+ OS: Ubuntu 18.04.2 LTS 64 bits
+ Python version 3.7.11
+ Pytorch version 1.7.1
+ CUDA Version 10.1

## Dataset:

[CWEB dataset](https://github.com/SimonHFL/CWEB)

## For GEC-PD system:

* Go to the gec-pseudo folder and carry out the following instructions

* Download all the required packedges and checkpoints from [GEC-PD](https://github.com/butsugiri/gec-pseudodata).

* Go to fairseq folder and install fairseq by:
```
cd fairseq
pip install --editable .
```
* Download generated positive and negative samples in [data](https://drive.google.com/drive/folders/1_DQ6bEXihB_BvLrY2_PXz3FfjLUixGv9?usp=sharing). 

* Fine-tune the model for CL using 1 GPU using the train.sh in train-scripts folder, please specify the path to your gec-pseudo folder and path to your binarized data folder.
```
chmod +x train.sh
./train.sh 0 model/test-cl
```
* Fine-tune the model for CL- using 1 GPU using the train-.sh in train-scripts folder, please specify the path to your gec-pseudo folder and path to your binarized data folder.
```
chmod +x train-.sh
./train-.sh 0 model/test-cl-
```
* Make prediction using predict.sh, for example:
```
./predict.sh 0 CWEB/data/tokenized/CWEB-G.test.tok.source G_3 model/test-cl/checkpoint3.pt output/test-cl
```
* Use [ERRANT toolkit](https://github.com/chrisjbryant/errant) to obtain the score, you should get the following result

| Method |Domain | Annotation | P | R | F0.5
| --- | --- | --- | --- | --- | ---
| CL- | S | 0 | 41.30 | 18.53 | 33.15
| CL- | S | 1 | 32.39 | 17.51 | 27.68
| CL- | G | 0 | 42.23 | 19.59 | 34.30
| CL- | G | 1 | 33.07 | 20.57 | 29.49
| CL | S | 0 | 41.48 | 21.44 | 34.94
| CL | S | 1 | 31.11 | 19.37 | 27.74
| CL | G | 0 | 42.41 | 23.01 | 36.29
| CL | G | 1 | 32.00 | 23.28 | 29.77

## For GEC-BART system:

* Go to the BART-GEC folder and carry out the following instructions

* Download all the required packedges and checkpoints from [GEC-BART](https://github.com/Katsumata420/generic-pretrained-GEC/tree/master/BART-GEC).

* Follow the instruction from [GEC-BART](https://github.com/Katsumata420/generic-pretrained-GEC/tree/master/BART-GEC) to train the model on BEA first.

* Go to GEC-BART folder and install fairseq by:
```
pip install --editable .
```
* Download generated positive and negative samples in [data](https://drive.google.com/drive/folders/1cKp5JnYXNIzgaCTqq6YVcnQjfi0dBDZl?usp=sharing).

* Fine-tune the model for CL using 4 GPUs using the train.sh in train-scripts folder, please specify the path to your BART-GEC folder, path to your trained BART model and path to your binarized data folder.
```
chmod +x train.sh
./train.sh 0,1,2,3 0.85 0.5 model/4gpu-cweb-0.85-0.5
```
* Fine-tune the model for CL- using 4 GPUs using the train-.sh in train-scripts folder, please specify the path to your BART-GEC folder, path to your trained BART model and path to your binarized data folder.
```
chmod +x train-.sh
./train-.sh 0,1,2,3 0.85 0.5 model/4gpu-cweb-0.85-0.5-
```
* Make prediction using translate-flexible-data.py. For example:
```
CUDA_VISIBLE_DEVICES=0 python3 translate-flexible-data.py --model_dir=model/4gpu-cweb-0.85-0.5 \
                --input_text=CWEB/data/tokenized/CWEB-S.test.tok.source \
                --output_dir=output/cweb-0.85-0.5-4gpu/S_18.txt \
                --checkpoint_file=checkpoint18.pt \
                --data_path=beam-sample-with-refined-counts/CWEB-3-data-bin
```

* Use [ERRANT toolkit](https://github.com/chrisjbryant/errant) to obtain the score, you should get the following result

| Method |Domain | Annotation | P | R | F0.5
| --- | --- | --- | --- | --- | ---
| CL- | S | 0 | 45.25 | 14.71 | 31.98
| CL- | S | 1 | 33.24 | 13.02 | 25.36
| CL- | G | 0 | 47.54 | 15.54 | 33.68
| CL- | G | 1 | 36.45 | 15.98 | 29.02
| CL | S | 0 | 46.98 | 16.26 | 34.10
| CL | S | 1 | 33.33 | 13.89 | 26.05
| CL | G | 0 | 50.00 | 16.79 | 35.82
| CL | G | 1 | 37.35 | 16.82 | 30.02

## Output

Our output result reported in the paper can be found in [output folder](https://github.com/nusnlp/geccl/tree/main/output). 

## Model Checkpoints

The fine-tuned checkpoints we obtained can be found in [GEC-BART-ckpt](https://drive.google.com/drive/folders/1U5tUI2SSipzETtcq5L_mXCiQweCiX3N-?usp=sharing) and [GEC-PD-ckpt](https://drive.google.com/drive/folders/1vY2duWRqXZSSWgxafk7354VM9iHlbQ1k?usp=sharing). 

## Citation

If you found our paper or code useful, please cite as:

```
@inproceedings{cao-etal-2021-grammatical-error,
    title = "Grammatical Error Correction with Contrastive Learning in Low Error Density Domains",
    author = "Cao, Hannan  and
      Yang, Wenmian  and
      Ng, Hwee Tou",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.419",
    pages = "4867--4874",
}
```
