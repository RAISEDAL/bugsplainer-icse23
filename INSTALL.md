> Disclaimer: All information of this file is also present is README.md.
> We recommend to solely follow the README.md file.

In order to replicate our test scores, a [CUDA](https://developer.nvidia.com/cuda-downloads)
supported GPU with at least 16 GB memory is needed.
In plain words, any NVIDIA GPU is CUDA supported.

Make sure you have python 3.9, CUDA 11.3 and the latest pip installed on your machine.
Then, to install `pytorch`, run
```shell
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0+cu113 torchtext==0.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

> If you want to run the Bugsplainer in CPU, remove `+cu113` part from all package versions.
> Note that we do not guaranty a proper replication on CPU as the tensor operations may
> vary based on CPU architecture.

Once pytorch is installed, install the remaining packages by running
```shell
pip install -r requirements.txt
```

You can run `python -m src.verify_installation` to validate the installation.
If it executes without any error, then all the necessary packages are installed.

Now, download one or all available model variants and data from
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7549218.svg)](https://doi.org/10.5281/zenodo.7549218).
Extract the downloaded file `bugsplainer.zip`.
Then replace the `data` directory of this repository with the extracted `data` directory
and `models` directory of this repository with the extracted `models` directory.

The subdirectory names inside the `models` directory are in format —
`{experiment_no}.{task}-{sub_task}-{input_len}-{output_len}-{batch_size}-{model_size}`.


Finally, to replicate Bugsplainer, run
```shell
python -m src.bugsplainer \
 --do_test \
 --model_name_or_path=models/262.finetune-sbt-random-512-64-16-60m/output/checkpoint-best-bleu \
 --config_name=models/config_60m.json \
 --tokenizer_name=Salesforce/codet5-base \
 --task="finetune" \
 --sub_task="sbt-random" \
 --desc="replicating-bugsplainer" \
 --data_dir=data \
 --max_source_length=512 \
 --max_target_length=64 \
 --eval_batch_size=16 \
 --cache_path=cache \
 --output_dir=output \
 --res_dir=result
```

For Windows Command Line, replace `\` with `^` for line continuation
```shell
python -m src.bugsplainer ^
 --do_test ^
 --model_name_or_path=models/262.finetune-sbt-random-512-64-16-60m/output/checkpoint-best-bleu ^
 --config_name=models/config_60m.json ^
 --tokenizer_name=Salesforce/codet5-base ^
 --task="finetune" ^
 --sub_task="sbt-random" ^
 --desc="replicating-bugsplainer" ^
 --data_dir=data ^
 --max_source_length=512 ^
 --max_target_length=64 ^
 --eval_batch_size=16 ^
 --cache_path=cache ^
 --output_dir=output ^
 --res_dir=result
```
