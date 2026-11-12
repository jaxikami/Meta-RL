# Retraining meta on Rorqual

Rorqual's compute nodes have no internet. The environment is built once on the
login node, and each job only runs it.

1. **Copy the code and the inputs.** Meta training needs `policy/saferl.pth`,
   `policy/saferl_best.pth` and `policy/apn_kinetic.pth`, and evaluation needs
   `results/log/saferl_log.json`. All four are gitignored, so a `git clone`
   does not bring them. Copy a bundle of the working tree to `$SCRATCH`
   instead:

   ```bash
   scp metarl_rorqual.tar.gz <user>@rorqual.alliancecan.ca:scratch/
   ssh <user>@rorqual.alliancecan.ca
   mkdir -p ~/scratch/Meta-RL && tar xzf ~/scratch/metarl_rorqual.tar.gz -C ~/scratch/Meta-RL
   ```

2. **Build the virtualenv** (once, on the login node):
   `cd ~/scratch/Meta-RL && bash cluster/rorqual/setup_env.sh`

3. **Set your allocation** in `train_meta.sbatch` (`--account=def-...`), then
   submit from the repository root: `sbatch cluster/rorqual/train_meta.sbatch`

The job runs the smoke test, trains meta, evaluates it against saferl with a
shared shield, and runs the z test. Outputs go to `policy/rorqual_<jobid>/` and
`results/{log,plot}/rorqual_<jobid>/`, and the log goes to
`metarl-meta-<jobid>.out`. `$SCRATCH` is purged over time, so copy the results
back when the job ends.
