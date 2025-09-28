# Cheatsheet - SLURM commands

## Check available partitions and their nodes.
-a: show all partitions
-l: long format
```console
sinfo -a -l
```
## Shows your own jobs
``` squeue --me
```
## Repeatedely execute a command with watch
``` watch squeue --me
```
## Running command using srun
`--time=01:00` specifies the runtime (in this case 1 minute).

```
srun --account=a-large-sc --time=1:00 -p debug --pty bash -c 'i=1; while [ $i -le 60 ]; do echo "Running on SLURM node... $i"; sleep 1; i=$((i+1)); done'
``` 
