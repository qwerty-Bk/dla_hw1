# ASR project 

## Installation guide

1. Installation

```shell
git clone -b final https://github.com/qwerty-Bk/dla_hw1.git
pip3 install -r dla_hw1/requirements.txt
```

2. Downloading the pretrained model
```shell
import gdown
gdown.download('https://drive.google.com/uc?id=10cW_aNW0H1VBFFyF2fR2aEnLct-sQBP1', 'cp.pth', True)
gdown.download('https://drive.google.com/uc?id=1oQwM4_QrxDIB4G2ZNWKVoB30vuVmdNhr', 'config.json', True)
```

3. Getting the results (in output.json file, the first lines are wer and cer of the novel before and after appyling the LM model, and the rest of the lines are the predictions)
```shell
python3 dla_hw1/test.py -c dla_hw1/hw_asr/configs/test_config.json -o output.json -r cp.pth
```


## Recommended implementation order

You might be a little intimidated by the number of folders and classes. Try to follow this steps to gradually undestand
the workflow.

1) Test `hw_asr/tests/test_dataset.py`  and `hw_asr/tests/test_config.py` and make sure everything works for you
2) Implement missing functions to fix tests in  `hw_asr\tests\test_text_encoder.py`
3) Implement missing functions to fix tests in  `hw_asr\tests\test_dataloader.py`
4) Implement functions in `hw_asr\metric\utils.py`
5) Implement missing function to run `train.py` with a baseline model
6) Write your own model and try to overfit it on a single batch
7) ~~Pain and suffering~~ Implement your own models and train them. You've mastered this template when you can tune your
   experimental setup just by tuning `configs.json` file and running `train.py`
8) Don't forget to write a report about your work
9) Get hired by Google the next day

## Before submitting

0) Make sure your projects run on a new machine after complemeting installation guide
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## TODO

These barebones can use more tests. We highly encourage students to create pull requests to add more tests / new
functionality. Current demands:

* Tests for beam search
* W&B logger backend
* README section to describe folders
* Notebook to show how to work with `ConfigParser` and `config_parser.init_obj(...)`
