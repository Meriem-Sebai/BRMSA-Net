# BRMSA-Net: Disclosing concealed colorectal polyps in colonoscopy images via Boundary Recalibration and Multi-Scale Aggregation Network
This repository contains the official Pytorch implementation of BRMSA-Net.

### Environment
- Create a virtual environment: `python -m venv BRMSANet`
- Activate the virtual environment: `source BRMSANet/bin/activate`
- Install the requirements: `pip install -r requirements.txt`

### Training
- Download MiT-b3's pretrained `weights` ([google drive](https://drive.google.com/drive/folders/1w59gNxY0z68XnJT4sHOiYM5lgy3tgE7g?usp=sharing).
- Run `sh train.sh` for training. 
### Testing
- We provide the trained weights
- Run `sh test.sh` for testing.
### Evaluation
- We provide the trained weights
- Run `sh test.sh` for testing.


### Citation
If you find this code useful in your research, please consider citing:

```
@article{Sebai2025BRMSANet,
  title     = {BRMSA-Net: Disclosing concealed colorectal polyps in colonoscopy images via Boundary Recalibration and Multi-Scale Aggregation Network},
  author    = {Sebai, Meriem and Goceri, Evgin},
  journal   = {Biomedical Signal Processing and Control},
  volume    = {110},
  pages     = {108083},
  year      = {2025},
  publisher = {Elsevier}
}
```


