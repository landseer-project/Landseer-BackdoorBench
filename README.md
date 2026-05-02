# Landseer-BackdoorBench

BackdoorBench fine-tuning (`ft`) / supervised fine-tuning (`sft`) post-training
defense integration for the [Landseer](https://github.com/landseer-project/Landseer)
ML security evaluation pipeline.

See `FT_SFT_DEFENSE.md` for the full integration design and usage notes.

## Layout

- `integrations/backdoorbench_post/` — wrapper container (`Dockerfile`, `Apptainer.def`, `main.py`)
- `configs/pipeline/backdoorbench_post_{ft,sft}.yaml` — example Landseer pipeline configs
- `scripts/upload_results_to_wandb.py` — push evaluation results to W&B
- `run_landseer.sh` — SLURM launcher
- `src/landseer_pipeline/...` — modifications to upstream Landseer required by this integration

## Quick start

```bash
cd integrations/backdoorbench_post
apptainer build --force backdoorbench_landseer_post.sif Apptainer.def

METHOD=ft  sbatch run_landseer.sh
METHOD=sft sbatch run_landseer.sh
```
