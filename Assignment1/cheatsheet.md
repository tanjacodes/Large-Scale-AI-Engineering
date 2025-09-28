# Cheatsheet - SLURM commands

## Check available partitions and their nodes.
-a: show all partitions
-l: long format
```
sinfo -a -l
```
## Shows your own jobs
```
squeue --me
```
## Repeatedely execute a command with watch
repeats squeue every 2s 
```
watch squeue --me
```
## Running command using srun
`--time=01:00` specifies the runtime (in this case 1 minute).
for accoutn specify your account, e.g. a06

```
srun --account=a06 --time=1:00 -p debug --pty bash -c 'i=1; while [ $i -le 60 ]; do echo "Running on SLURM node... $i"; sleep 1; i=$((i+1)); done'
```

## Interactive session
define environment
```
srun --account=a06 --container-writable --environment=my_env -p debug --pty bash
```


## Submit a job

```
sbatch my_script.sh  # submit job
```

### Container Build

Get interactive shell with container support, build container with podman, convert to enroot format.

```bash
srun --container-writable -p debug --pty bash
podman build -t my_image:tag .  # Creates the actual container image
enroot import -o ~/scratch/my_image.sqsh podman://my_image:tag
```

## TOML Config

```toml
# Specifies the path to your container image file .sqsh in scratch which will be executed
image = "~/scratch/my_image.sqsh"

# Makes CSCS storage systems available inside the container:
# - capstor: High-capacity storage for large datasets
# - iopsstor: High-IOPS storage for performance-sensitive workloads
# - users: Home directories and user data
mounts = ["/capstor", "/iopsstor", "/users"]

```

### Batch Header
job-name: name of the job, so if you check for squeue --me, you can identify your job by name
time: in example 1 hour
output: path to output file
error: path to error file, to get error outputs
```bash
#!/bin/bash
#SBATCH --job-name=name
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --environment=my_env
#SBATCH --output=~/scratch/job.out
#SBATCH --error=~/scratch/job.err
```

## Quick Checks

```bash
quota        # check usage quota (on ela)
nvidia-smi   # see GPUs
```
