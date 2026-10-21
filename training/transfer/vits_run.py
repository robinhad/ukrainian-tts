"""Train VITS in stateful 5K stages, with fixed-panel evaluation after each stage."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(root):
    record=json.loads((root/'experiment.json').read_text())
    for step in range(record['evaluation_interval'],record['steps']+1,record['evaluation_interval']):
        command=[sys.executable,'-m','training.transfer.vits','train','--root',str(root),'--epochs',str(step//1000)]
        if step>record['evaluation_interval']:command.append('--resume')
        subprocess.run(command,check=True)
        if step==record['evaluation_interval'] and (root/'baseline_events').exists():
            shutil.copytree(root/'baseline_events',root/'train_model/tensorboard',dirs_exist_ok=True)
        subprocess.run([sys.executable,'-m','training.transfer.vits_evaluate','--root',str(root),'--step',str(step)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)


if __name__=='__main__':main()
