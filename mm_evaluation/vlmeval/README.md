
## Installation

```bash
pip install -r requirements.txt
```

## Eval

```bash
# eval multiple models on multiple benchmarks 
bash submit_eval.sh "OCRBench MMStar MMMU_DEV_VAL ChartQA_TEST DocVQA_VAL HallusionBench ScienceQA_TEST MathVista_MINI TextVQA_VAL AI2D_TEST InfoVQA_VAL RealWorldQA MMVet MME"  "Eagle2-1B Eagle2-2B" work_dirs/eagel2-1b test 8

bash submit_eval.sh "OCRBench"  "Eagle2-1B Eagle2-2B" work_dirs/eagel2-1b test 8
```