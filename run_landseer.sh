#!/bin/bash
#SBATCH --job-name=landseer_bb
#SBATCH --output=landseer_%j.out
#SBATCH --error=landseer_%j.err
#SBATCH --mail-user=wang7395@purdue.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account=zghodsi
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -eo pipefail

METHOD="${METHOD:-ft}"
ATTACK_CONFIG="${ATTACK_CONFIG:-configs/attack/test_config_2.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-landseer-backdoorbench}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-backdoorbench-post}"
WANDB_TAGS="${WANDB_TAGS:-gilbreth backdoorbench ${METHOD}}"
FORCE_REBUILD_IMAGE="${FORCE_REBUILD_IMAGE:-0}"
PROJECT_ROOT="/scratch/gilbreth/${USER}/Landseer"
INTEGRATION_DIR="${PROJECT_ROOT}/integrations/backdoorbench_post"
APPTAINER_IMAGE="${INTEGRATION_DIR}/backdoorbench_landseer_post.sif"
APPTAINER_DEF="${INTEGRATION_DIR}/Apptainer.def"
RUN_RESULTS_ROOT="${PROJECT_ROOT}/results"
RUN_LOG_ROOT="${PROJECT_ROOT}/logs"
CACHE_ROOT="${PROJECT_ROOT}/cache"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/gilbreth/wang7395/.conda/envs/2025.06-py313/landseer}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
LANDSEER_BIN="${LANDSEER_BIN:-${CONDA_ENV_PATH}/bin/landseer}"

case "${METHOD}" in
  ft)
    PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/pipeline/backdoorbench_post_ft.yaml}"
    ;;
  sft)
    PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/pipeline/backdoorbench_post_sft.yaml}"
    ;;
  *)
    echo "Unsupported METHOD='${METHOD}'. Use ft or sft." >&2
    exit 2
    ;;
esac

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

export APPTAINER_CACHEDIR="/scratch/gilbreth/${USER}/apptainer_cache"
export APPTAINER_TMPDIR="/scratch/gilbreth/${USER}/apptainer_tmp"
export WANDB_DIR="/scratch/gilbreth/${USER}/wandb/${SLURM_JOB_ID:-manual}"
export WANDB_CACHE_DIR="/scratch/gilbreth/${USER}/.cache/wandb"
export WANDB_CONFIG_DIR="/scratch/gilbreth/${USER}/.config/wandb"
export XDG_CACHE_HOME="/scratch/gilbreth/${USER}/.cache"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${APPTAINER_CACHEDIR}" \
  "${APPTAINER_TMPDIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${XDG_CACHE_HOME}"

cd "${PROJECT_ROOT}"

echo "========== Landseer BackdoorBench Run =========="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Method: ${METHOD}"
echo "Pipeline config: ${PIPELINE_CONFIG}"
echo "Attack config: ${ATTACK_CONFIG}"
echo "Conda env: ${CONDA_ENV_PATH}"
echo "Python bin: ${PYTHON_BIN}"
echo "Landseer bin: ${LANDSEER_BIN}"
echo "Apptainer image: ${APPTAINER_IMAGE}"
echo "WANDB project: ${WANDB_PROJECT}"
echo "==============================================="

if [[ ! -f "${APPTAINER_IMAGE}" || "${FORCE_REBUILD_IMAGE}" == "1" ]]; then
  echo "Building Apptainer image for BackdoorBench wrapper..."
  rm -f "${APPTAINER_IMAGE}"
  (
    cd "${INTEGRATION_DIR}"
    apptainer build --force "${APPTAINER_IMAGE}" "$(basename "${APPTAINER_DEF}")"
  )
else
  echo "Reusing existing Apptainer image: ${APPTAINER_IMAGE}"
fi

if ! ${PYTHON_BIN} -c "import wandb" >/dev/null 2>&1; then
  echo "wandb not found in active environment; installing into current conda env..."
  ${PYTHON_BIN} -m pip install wandb
fi

before_snapshot=$(mktemp)
after_snapshot=$(mktemp)
find "${RUN_RESULTS_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort > "${before_snapshot}" || true

${LANDSEER_BIN} \
  --config "${PIPELINE_CONFIG}" \
  --attack-config "${ATTACK_CONFIG}" \
  --results-dir "${RUN_RESULTS_ROOT}" \
  --log-dir "${RUN_LOG_ROOT}"

find "${RUN_RESULTS_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort > "${after_snapshot}" || true
new_run_dir=$(comm -13 "${before_snapshot}" "${after_snapshot}" | tail -n 1)
rm -f "${before_snapshot}" "${after_snapshot}"

if [[ -z "${new_run_dir}" ]]; then
  new_run_dir=$(find "${RUN_RESULTS_ROOT}" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' | sort -n | tail -n 1 | awk '{print $2}')
fi

if [[ -z "${new_run_dir}" || ! -d "${new_run_dir}" ]]; then
  echo "Failed to determine Landseer run directory" >&2
  exit 3
fi

echo "Latest Landseer run directory: ${new_run_dir}"

wandb_args=(
  --run-dir "${new_run_dir}"
  --project "${WANDB_PROJECT}"
  --group "${WANDB_GROUP}"
  --tags ${WANDB_TAGS}
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  wandb_args+=(--entity "${WANDB_ENTITY}")
fi

${PYTHON_BIN} scripts/upload_results_to_wandb.py "${wandb_args[@]}"

echo "Landseer + BackdoorBench ${METHOD} run completed successfully."
