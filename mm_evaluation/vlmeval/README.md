
## Installation

```bash
pip install -r requirements.txt
```

## Eval

```bash
cd mm_evaluation/vlmeval

export OPENAI_API_BASE=TBD
export OPENAI_API_KEY=sk-TBD

export MASTER_PORT=$((RANDOM % 101 + 20000))
torchrun --nproc-per-node=8 --master_port $MASTER_PORT run.py \
         --data MMBench_DEV_EN_V11 OCRBench MMMU_DEV_VAL MMStar ChartQA_TEST DocVQA_VAL HallusionBench ScienceQA_TEST TextVQA_VAL AI2D_TEST InfoVQA_VAL RealWorldQA MMVet MME \
         --model InstructVLA \
         --work-dir path/to/InstructVLA/outputs/vlmeval/InstructVLA \
         --tag results \
         --model_path path/to/model.pt \
         --reuse \
         --verbose
```