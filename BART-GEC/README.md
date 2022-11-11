# BART-GEC
- Fine-tuned BART model for GEC.
  - Only English GEC.
- Ensemble of different pretrained models of BART.
- This script is based on fairseq.
  - original commit id: 9f4256e [[link]](https://github.com/pytorch/fairseq/tree/9f4256edf60554afbcaadfa114525978c141f2bd)
  - fairseq README: `FAIRSEQ_README.md`
- This files run on a linux server

## How To Run
1. Prepare the BEA-train/valid/test data (Lang-8, NUCLE, and so on).
    - https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
2. Prepare pretrained BART model (`bart.large.tar.gz`) and related files
 (`encoder.json`, `vocab.bpe` and `dict.txt`).
    - `bart.large.tar.gz`: [url](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
    - `encoder.json`: [url](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json)
    - `vocab.bpe`: [url](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe)
    - `dict.txt`: [url](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt)
3. unzip the bart.large.tar.gz
4. Download data from
    - [url]https://drive.google.com/file/d/1AFdFaiT9J0BGmListcNP143cMT8udHq6/view?usp=sharing
4. Download the 4 checkpoints from OneDrive link and move the files into bart.large folder
    - [url](https://nusu-my.sharepoint.com/:f:/g/personal/e0406577_u_nus_edu/EqVOiU-QjQBHpM5bvl2wLSMBLwDPDDHrADUV7FB7FEMLxw)
    - rename the checkpoint you want to use as checkpoint_best.pt
5. Run translate_ensemble.sh to get the ensemble outputs
6. Run translate.sh to get individual outputs
7. Run ensemble_average_checkpoints.py on the 4 checkpoints to get the average checkpoint

## To perform full training
1. Follow steps 1-3 above
2. Apply BART BPE to the BEA-train/valid data.
    - Use `apply_bpe.sh`.
3. Binarize train/valid data.
    - Use `preprocess.sh`.
4. Fine-tune the BART model with binarized data.
    - Use `train.sh`.
5. Translate BEA-test using the fine-tuned model.
    - Use `translate.sh`.



