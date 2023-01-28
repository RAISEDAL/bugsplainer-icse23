# Replication Package of Bugsplainer

## Explaining Software Bugs Leveraging Code Structures in Neural Machine Translation

#### Accepted Papers at ICSE 2023


> Parvez Mahbub, Ohiduzzaman Shuvo, and M. Masudur Rahman. Explaining Software Bugs Leveraging Code Structures in Neural Machine Translation. In Proceeding of The 45th IEEE/ACM International Conference on Software Engineering (ICSE 2023), pp. 12, Melbourne, Australia, May 2023 (To appear).

## Abstract
Software bugs claim approximately 50% of development time and cost the global economy billions of dollars. Once a bug is reported, the assigned developer attempts to identify and understand the source code responsible for the bug and then corrects the code. Over the last five decades, there has been significant research on automatically finding or correcting software bugs. However, there has been little research on automatically explaining the bugs to the developers, which is essential but a highly challenging task. In this paper, we propose Bugsplainer, a transformer-based generative model, that generates natural language explanations for software bugs by learning from a large corpus of bug-fix commits. Bugsplainer can leverage structural information and buggy patterns from the source code to generate an explanation for a bug. Our evaluation using three performance metrics shows that Bugsplainer can generate understandable and good explanations according to Google's standard, and can outperform multiple baselines from the literature. We also conduct a developer study involving 20 participants where the explanations from Bugsplainer were found to be more accurate, more precise, more concise and more useful than the baselines.

## File Structure
- `data`: contains the data-files
- `models`: contains the model checkpoints. 
  Each subdirectory has the format `<checkpoint-name>/output/checkpoint-best-bleu`.
- `output`: contains sample explanations from Bugsplainer, Bugsplainer 220M,
  and the baselines.
- `runs`: contains the output of each run during the replication.
  Each subdirectory is named after the `--desc` parameter discussed below.
- `scrape_github`: python module that we used to scrape bug-fix commits from GitHub.
  Simply run `python __main__.py` to scrape a new dataset.
- `src`:
  - `bugsplainer`: necessary scripts for replication
  - `compute_scores_from_explanations.py`: computes BLEU score, Semantic Similarity,
    and Exact Match for the generated explanations. If no argument given, it computes
    these scores from the sample from `output/run4` directory. To evaluate pyflakes
    specifically (using max score scheme described in the paper), run as 
    `python -m src.compute_scores_from_explanations pyflakes`.
  - `explore_scores.py`: Performs statistical tests for the metric scores in 5 runs.
- `requirements.txt`: the requirements file for `pip`.

## Replicate
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

You can replace the name of the model subdirectory for parameter
`--model_name_or_path` to replicate the score for different experiment.
In such a case, you also need to change the value of `--sub_task` parameter
encoded in the subdirectory name.
The subdirectory names are in format —
`{experiment_no}.{task}-{sub_task}-{input_len}-{output_len}-{batch_size}-{model_size}`.
The mapping from our experiment name to the checkpoint name is as follows.

| RQ  | Experiment                     | Checkpoint                              |
|-----|--------------------------------|-----------------------------------------|
| 1   | Bugsplainer                    | 262.finetune-sbt-random-512-64-16-60m   |
| 1   | Bugsplainer Cross-project      | 264.finetune-sbt-project-512-64-16-60m  |
| 1   | Bugsplainer 220M               | 268.finetune-sbt-random-512-64-16-220m  |
| 1   | Bugsplainer 220M Cross-project | 270.finetune-sbt-project-512-64-16-220m |


### Compute Scores

Once you have the generated explanations, you can compute their metric scores by running

```shell
python -m src.compute_scores_from_explanations <desc>
```
where `<desc>` should be replaced with the value of `--desc` flag in the previous step.

## Training Details
We validate the model after each epoch by BLEU score.
Training runs for 100 epochs with early stopping.
We use the model with the best result on the validation set as our final model.
We then compute the average BLEU score on the test set.
We call this whole process a _run_.
In RQ3, we report the average score of five such runs with different random initialization.

### Hyper-parameters

The hyperparameters used during the experiment are listed below


| Hyper-parameter         | Value  |
|-------------------------|--------|
| Optimizer               | AdamW  |
| Adam Epsilon            | 1e-8   |
| Beam Size               | 10     |
| Batch Size              | 16     |
| Learning Rate           | 5e-5   |
| Input Length            | 512    |
| Output Length           | 64     |
| Epoch                   | 100    |
| Early Stopping Patience | 5      |
| Learning Rate Scheduler | Linear |
| Warmup Steps            | 100    |

The model-specific hyperparameters can be found in the config files
in the `models` directory (download from Zenodo).

--- 
[![License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
<br/>
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

> Something not working as expected? Create an issue [here](https://github.com/RAISEDAL/bugsplainer-icse23/issues).