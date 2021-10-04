#!/bin/bash
# This script runs a slurm job array with one job for each of the sweep specs
# produced by python file OVERRIDES_GENERATOR.
# Usage:
#     $ bash openmind_launch.sh path/to/my/overrides/generator.py
# Before running, please check that the time limit and memory requirement in
# openmind_task.sh are suitable.

# OVERRIDES_GENERATOR is the python script that generates the config overrides.
# If is an optional argument to this bash script, with the following default:
OVERRIDES_GENERATOR=${1:-"sweeps/lif_rnn.py"}

# Check to make sure OVERRIDES_GENERATOR exists
if [[ ! -f $OVERRIDES_GENERATOR ]]; then
    echo "The file $OVERRIDES_GENERATOR does not exist"
    exit
fi

# Generate array of config overrides by running $OVERRIDES_GENERATOR
overrides_array=()
while read line ; do
  # First make sure '\n' is not in the config ovverrides. If it is that would
  # cause problems below because we serialize overrides_array with '\n' as a
  # separator.
  if [[ "$line" == *"$\n"* ]]; then
    echo $line
    echo "config overrides shown above contains '\n'. This is not allowed."
    exit
  fi
  # Append the line to overrides_array
  overrides_array+=("$line")
done < <(python ${OVERRIDES_GENERATOR})

# Config name is first element of overrides_array, so pop that out
config_name=${overrides_array[0]}
overrides_array=("${overrides_array[@]:1}")

# Get size of overrides_array, i.e. number of jobs to launch. Must subtract 1
# because of zero-indexing in the sbatch array launch below.
num_job_array=$((${#overrides_array[@]} - 1))

# Get user confirmation to launch the array
echo "Ready to submit $((${num_job_array} + 1)) jobs? \
Press 'y' and Enter to continue."
read confirmation
if [[ ! $confirmation =~ ^[Yy]$ ]]
then
  echo "Launch canceled."
  exit
fi

# Bash cannot pass array as an argument and sbatch --array cannot pass elements
# of a bash array as arguments do the different jobs in an array, so our only
# option is to serialize the entire array of overrides and pass that to
# openmind_task.sh, then in openmind_task.sh it is de-serialized and indexed.
overrides_serialized="$( printf "\n%s" "${overrides_array[@]}" )"

# Submit job array to SLURM
array_launch=$(\
    sbatch --array=0-$num_job_array openmind_task.sh \
    $config_name "$overrides_serialized"
    )
job_id=${array_launch##*' '}
echo "Launched job ${job_id}"

# Create directory for logs
mkdir -p logs/${job_id}
mkdir -p logs/${job_id}/slurm_logs

# Create cancel commands file
cancel_commands_file=logs/${job_id}/cancel_commands.sh
echo "" > ${cancel_commands_file}

# Write config  name to a file
echo ${config_name} > logs/${job_id}/config_name.log

# Write overrides generator name to a file
echo ${OVERRIDES_GENERATOR} > logs/${job_id}/overrides_generator.log