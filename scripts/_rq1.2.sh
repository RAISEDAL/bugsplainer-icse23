python -m src.bugsplainer \
 --do_test \
 --model_name_or_path=models/264.finetune-sbt-project-512-64-16-60m/output/checkpoint-best-bleu \
 --config_name=models/config_60m.json \
 --tokenizer_name=Salesforce/codet5-base \
 --task="finetune" \
 --sub_task="sbt-project" \
 --desc="replicating-bugsplainer-2" \
 --data_dir=data \
 --max_source_length=512 \
 --max_target_length=64 \
 --eval_batch_size=16 \
 --cache_path=cache \
 --output_dir=output \
 --res_dir=result

python -m src.compute_scores_from_explanations "replicating-bugsplainer-2"
