# English Indic Transformer

This repository contains the code for machine translation transformer in pytorch for English to Hindi. This project is
for educational purpose only for me to learn properly about transformers. The intention of this project is to understand the implementation of transformer architecture for machine translation, learn and implement the transformer architecture from scratch in pytorch and get a hands-on experience of the working of various components of transformer architecture such as multi-head attention, positional encoding, feed forward neural networks etc. Also, learn how cross attention works in the decoder part of the transformer along with key-value cahcing and quantization techniques to optimize the model for faster inference.

**_HF repository: [smathad/eng-indic-transformer](https://huggingface.co/smathad/eng-indic-transformer)._**

**Note**

The model present in HF directory is currently trained for 10 epochs on English to Hindi dataset with **~6 BLEU**. Report can be found [here](https://huggingface.co/smathad/eng-indic-transformer/tree/main/reports).

### Cloning the repository:

```bash
# https
git clone https://github.com/sameera-g-mathad/eng-indic-transformer.git

# ssh
git clone git@github.com:sameera-g-mathad/eng-indic-transformer.git

```

### Virtual Environment Setup:

```bash
# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate  # On Mac

```

### Install Dependencies:

```bash

# Requirements installation
make install

# Development requirements installation
make install_dev

```

### Env file setup:

Create a `.env` file in the root directory of the repository and add the following variables:

```
HF_TOKEN=<your_huggingface_token>
```

### Datasets:

The datasets used for training the model are as follows:

- Hindi dataset: [damerajee/english-to-hindi-l](https://huggingface.co/datasets/damerajee/english-to-hindi-l).
- Kannada dataset: [damerajee/en-kannada](https://huggingface.co/datasets/damerajee/en-kannada).

### Training the model:

To train the model, run notebooks in the `notebooks/train` folder sequentially.

- `(1)_download_data.ipynb` : Contains code to download the dataset, for both English-Hindi and English-Kannada language pairs. The datasets be downloaded into `/data` folder.
- `(2)_dataset.ipynb`: Contains code to train the tokenizer using SentencePiece library. **Optional** as pre-trained tokenizer files are provided in the `tokenizers` folder in the hf directory.
- `(3)_hf_pull.ipynb`: Contains code to pull the pre-trained tokenizer, model, optimizer and reports from the HF repository highlighted above. All the cells can be run sequentially to get the pre-trained model and tokenizer. Tokenizer files are saved in the `/tokenizers` folder and model in `/models` and optimizer checkpoints are saved in the `/checkpoints`, reports in `/reports` folder.
- `(4)_en_hindi_train.ipynb`: Contains code to train the transformer model for machine translation. The model is trained using the pre-trained tokenizer and dataset downloaded in the previous steps. The trained model checkpoints are saved in the `/checkpoints` folder.
- `(5)_hf_push.ipynb`: Contains code to push the trained model, tokenizer, optimizer and reports to the HF repository highlighted above. Also, contains piece of code to tag the commit for versioning.

### Inference:

To perform inference using the trained model, run the notebook in the `notebooks/inference` folder.

- `(1)_hf_inference_pull.ipynb`: Contains code to pull the trained model and tokenizer from the HF repository highlighted above. The tokenizer files are saved in the `/tokenizers` folder and model in `/models` folder.
- `(2)_inference.ipynb`: Contains code to perform inference using the trained model and tokenizer. The input sentences can be provided in the code cell to get the translations.

### `/en-inic-transformer` folder structure:

This folder contains the implementation of the transformer architecture for machine translation using pytorch. The folder structure is as follows:

- `components.py`: Contains the implementation of various components of the transformer architecture such as multi-head attention, positional encoding, feed forward neural networks etc. Also contains code for key-value caching. **Need to implement quantization optimization.**
- `model.py`: Contains the code to train the transformer model for machine translation using the components implemented in `components.py`.
- `processing.py`: Contains the code to preprocess the dataset for training the transformer model, such as DataLoader, collate functions etc.
- `tokenizer.py`: Contains the code to train the tokenizer using SentencePiece library.

### Results (English to Hindi Translation so far):

Results after training for 15 epochs on English to Hindi dataset.

```csv
train_loss,test_loss,bleu
5.423146942715729,4.58064692665196,1.318473518518219
4.246883749776191,3.993251299151192,2.703825917164902
3.788441771042956,3.69056556888882,3.95213225322378
3.54239963612081,3.534704078496949,4.585495794121761
3.389366801887802,3.432832670468847,5.069833865415286
3.279219676778366,3.348437811807802,5.344551214962391
3.193550752958385,3.302223570263289,5.5763067471930485
3.124565699554557,3.2509325661320787,5.712103258928199
3.0678364998554204,3.221580374080216,5.80111526889886
3.019255864361234,3.19820795995313,5.957311082953983
2.976692221355697,3.160101642313029,6.062638003974348
2.9383026494880267,3.1474825345388107,6.1684363013209484
2.9040966032067024,3.1229404701805286,6.198855989956427
2.874136930282569,3.109688318986027,6.206940691481408
2.8463363223889733,3.100018622675996,6.350329949680979
```

### References:

- Sebastian Raschka, “Build a Large Language Model (From Scratch)”.
- (Youtube video - Aladdin Persson) [Pytorch Transformers from Scratch (Attention is all you need)](https://www.youtube.com/watch?v=U0s0f995w14&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=40).
