# Position Bias Analyzer
## Quick Start

### Environment Setup
Create a conda environment. Python version should be >= 3.8. We here use Python 3.9 to ensure no potential conflicts.
```bash
conda create -n Position_Bias_Analyzer python=3.9
conda activate Position_Bias_Analyzer
```

Clone the github repositiory and cd to the root.

Install dependencies:
```bash
pip install -e .
```

Export Python path and your API keys.
```bash
export PYTHONPATH=$PWD
export KEYS=path/to/keys.json
```

For API keys, store than in a `json` file (e.g., `keys.json`) using the following format:
```json
{
    "claude-keys":[
        "key1",
        "key2"
    ],
    "openai-keys": [
        "key1",
        "key2"
    ],
    "gemini-keys":[
        "key1",
        "key2"
    ],
}
```
If you just want to use some of the LLMs as judge models, you may only include the API keys for them and ignore the others.


**Now you can move on to use LLM as judges to evaluate Model answer pairs**.

**If you have already prepared the LLM judgments, you may directly jump to the analysis part**.


### Subjective Evaluation using LLM Judges
We sample a small set of MTBench questions [here](./QA_data/example/MTBench_data_sliced.xlsx) for this demo. Note that data preperation format for MTBench (multi-run QA) and DevBench (single-run QA) are different. For more information, please visit [this session](#data-preparation-for-llm-as-a-judge).

Run the following script:
```bash
chmod +x ./scripts/run_judgment/run_MTBench_example.sh
./scripts/run_judgment/run_MTBench_example.sh
```
or

```bash
python3 subeval/subjective/sub_eval.py --data QA_data/example/MTBench_data_sliced.xlsx --benchmark MTBench --model gpt-4 claude-v1 --refm vicuna-13b-v1.3 --judge gemini-pro --eval-nopt 3 --eval-proc 1 --mode dual

python3 subeval/subjective/sub_eval.py --data QA_data/example/MTBench_data_sliced.xlsx --benchmark MTBench --model gpt-4 claude-v1 --refm vicuna-13b-v1.3 --judge claude-3-haiku-20240307  --eval-nopt 3 --eval-proc 1 --mode dual

python3 subeval/subjective/sub_eval.py --data QA_data/example/MTBench_data_sliced.xlsx --benchmark MTBench --model gpt-4 claude-v1 --refm vicuna-13b-v1.3 --judge gpt-3.5-turbo-1106  --eval-nopt 3 --eval-proc 1 --mode dual
```

Here are the explanation for the arguments:
- `--data`: the relative path to the QA data, prepared following specific format. Preferred to use `xlsx`.
- `--benchmark`: the benchmark name. Now only supports MTBench or DevBench. For more datasets, please revise the [subeval](./subeval/subjective) module, including prompt settings and number of QAs to fit specific needs.
- `--model`: the answer-generating Models to evaluate. The exact column name should be included in the columns of `--data` file following specific format.
- `--refm`: the reference/baseline model for comparison. All the `--model`s will be compared to this reference model, excluding exact matches. Therefore, you may either include or exclude `--refm` in `--model`
- `--judge`: the Judge model. Please input the official model name.
- `--eval-nopt`: the **Option Mode**, should be 2 or 3. For Option-2 mode, LLM will choose A (response 1 is better than response 2) or B (response 2 is better than response 1). Option-3 mode adds an option C (tie).
- `--eval-proc`: the number of process to run simultaneously. If cannot run successfully, please turn it to 1.
- `--mode`: should be "dual" because we are investigating position bias, and "dual" includes the pair of queries with swapped-order Model responses. However for pure evaluation purpose, you may also choose "random" such that the Model response order is randomly assigned.


## Position Bias Analysis
The analysis functions are programmed in [position_bias_analyzer](./position_bias_analyzer/), which includes the calculation for position consistency, preference fairness, lengths, win rates, and judge agreement.


## Data Preparation for LLM-as-a-Judge
The columns and requirements are specified below:

### MTBench (2-round QA)
The columns for 2-round QA datasets like MTBench should be prepared to have the following columns:
1. `question_1`: the first question content in a single string.
2. `question_2`: the second question content in a single string.
3. `index`: the question ID, should be unique.
4. `evaluating_guidance`: may keep empty if do not have such; however, you must include this column.
5. `task`: the Task of the question.
6. `reference_answer_1`: some questions will include a reference answer while some will not. If there's no reference answer, keep an empty column; otherwise should be filled with reference answer to `question_1`.
7. `reference_answer_2`: similarly for `question_2`.
8. `answer_1-{model_id}`: the response of the `Model` on `question_1`.
9. `answer_2-{model_id}`: the response of the `Model` on `question_2`.


### DevBench (1-round QA)
The columns for 1-round QA datasets like DevBench should follow the convention for DevBench, including the following columns:
1. `questions`: the task question
2. `task`: the Task. Different evaluating metrics on the same task can be considered different `Task`s in our context. For example, in DevBench, `UML_class` is a task name, while there are two evaluating metrics (`general` and `faithfulness`). One may consider `UML_class-general` and `UML_class-faithfulness` as different Tasks in our context.
3. `index`: for DevBench, this is `[project_name]_[task]`. You may use whatever question ID you want - just to make it unique.
4. `evaluating_guidance`: the evaluation metrics, such as `general` and `faithfulness`. In our study, we seperate the `general` metrics into more detailed ones.
5. `reference_answer`: should be empty if not available, but make sure to keep the empty column.
6. `answer-{model_id}`: the response of `Model` on the `questions`.


## Data Preparation for Judging the Judges:
After running LLM-as-a-Judge evaluations, the results will be stored in a directory under the `output` folder. **Please keep the name of resulting files unchanged**. Feel free to move the resulting directory around and revise the paths accordingly for analysis functions.


## Acknowledgment
We make use of DevBench's code for LLM judge (subjective evaluation). Thank them for their clear and useful code. Their original codes can be accessed from [here](https://github.com/open-compass/DevBench/tree/main/llm_judge).



