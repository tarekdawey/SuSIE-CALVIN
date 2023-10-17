export DIFFUSION_MODEL_CHECKPOINT=<path_to_checkpoint>
export GC_POLICY_CHECKPOINT=<path_to_checkpoint>
export NUM_EVAL_SEQUENCES=10

python calvin_models/calvin_agent/evaluation/evaluate_policy_subgoal_diffusion.py --dataset_path mini_dataset --custom_model
