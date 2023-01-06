# CroMIC-QA: The Cross-Modal Information Complementation based Question Answering
This repository is the offical implementation of CroMIC-QA: The Cross-Modal Information Complementation based Question Answering.

## Requirements
To install requrements:

    conda install --yes --file requirements.txt

## Data
All the data is released [here](https://drive.google.com/file/d/1J06guaxvcWrbSbpfOPQoi6QyRCitW3Jl/view?usp=sharing).

## Data Process

### Text
1. Word Segmentation by Jieba

2. Word Embeddings from https://ai.tencent.com/ailab/nlp/zh/embedding.html


### Image
1. Image Vectorization Backbones: tf.keras.applications.MobileNetV2/ResNet50V2/Xception.

## Training
To train answer re-ranking models in the paper, setting parameters in config.py, run this command:

    python train.py

To fine-tune the image models, setting parameters in config.py, run this command:

    python train_v.py

## Evaluatation
To evaluate answer re-ranking models, run this command:

    import train
    train.evaluate()
To evaluate fined-tuned image models, run this commmand:

    import train_v
    train_v.evaluate()

## Pre-trained Models
[Pre-trained answer re-ranking model](model_weights/Rank_datatype6_concat_MN_FT.h5) trained on data type all with 'concat' fusion strategy and image embedding obtained by fine-tuned MobileNetV2 model.

## Results

CroMIC-QA Answer Re-ranking Model with 'concat' fusion strategy and  image embedding obtained by fine-tuned MobileNetV2 model.
| Data type | MAP | MRR | P@1 | P@3 | OPA
| - | - | - | - | - | - |
| 1 | 0.8738 | 0.8971 | 0.8105 | 0.7157 | 0.8764 |
| 2 | 0.8897 | 0.9048 | 0.8243 | 0.7537 | 0.8951 |
| 3 | 0.8966 | 0.9180 | 0.8475 | 0.7673 | 0.8937 |
| all | 0.9164 | 0.9348 | 0.8783 | 0.7715 | 0.9191 |

## License
This project is released under the [MIT License](LICENSE).