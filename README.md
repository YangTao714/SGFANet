# SGFANet

This repository contains the official implementation of **SGFANet**, a deep learning model for image super-resolution.

The code is built on a BasicSR-style training and testing pipeline, with configuration files for x2, x3, and x4 super-resolution experiments.

## News

The related paper is currently under submission. Experimental data, pretrained models, detailed results, and additional materials will be released after the paper review decision is available.

## Project Structure

```text
basicsr/        Core model, data, training, testing, loss, and utility code
options/        Training and testing configuration files
requirements.txt
VERSION
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train SGFANet from scratch:

```bash
python basicsr/train.py -opt options/train/train_SGFANet_x2_scratch.yml
```

Test SGFANet:

```bash
python basicsr/test.py -opt options/test/test_CATANet_x2.yml
```

Please update the dataset paths and pretrained model paths in the corresponding YAML files before running training or testing.

## Acknowledgement

This project follows the BasicSR-style code structure. We thank the open-source community for useful research codebases and tools.

## Citation

Citation information will be added after the paper is publicly available.
