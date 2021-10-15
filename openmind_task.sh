#!/bin/bash
# This script is a slurm job script. It should be not be run itself, but instead
# only called from openmind_launch.sh as part of a job array.
# Be sure to check that the time, memory, and partition are suitable for your
# task.

#SBATCH -o ./logs/%A/slurm_logs/%a.out
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --partition=jazayeri

CONFIG_NAME=$1
OVERRIDES_SERIALIZED=$2

# Split $OVERRIDES_SERIALIZED into an array by '\n'
IFS=$'\n' read -rd '' -a overrides_array <<<"$OVERRIDES_SERIALIZED"

# Get config overrides for current task ID in the job array
config_overrides=${overrides_array[${SLURM_ARRAY_TASK_ID}]}
echo "config_overrides: ${config_overrides}"

# Try to extract log_dir entry in config_overrides to include as both a comment
# in the cancellation command and as the --log_directory flag for main.py
log_dir_detector="log_dir="
if [[ "$config_overrides" == *${log_dir_detector}* ]]; then
  log_dir=${config_overrides#*${log_dir_detector}}
  log_dir=${log_dir%%"\""*}
else
  log_dir="${SLURM_ARRAY_TASK_ID}"
fi

# Write cancellation command to file
cancel_commands_file=logs/${SLURM_ARRAY_JOB_ID}/cancel_commands.sh
echo "scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  # $log_dir" >> \
${cancel_commands_file}

# Make metadata to associate slurm task ID
metadata="Task ID: ${SLURM_ARRAY_TASK_ID}"

# Print the full log directory
full_log_dir="logs/${SLURM_ARRAY_JOB_ID}/${log_dir}"
echo "full_log_dir: ${full_log_dir}"

# Run main.py
module load openmind/singularity
python3 main.py \
--config="${CONFIG_NAME}" \
--config_overrides="${config_overrides}" \
--log_directory="${full_log_dir}" \
--metadata="${metadata}"