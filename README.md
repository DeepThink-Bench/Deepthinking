<h1 align="center">
<br>
Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs
</h1>
<p align="center">
  <a href="https://deepthink-bench.github.io/DeepThink-Bench/"><b>[🌐 Website]</b></a> •
  <a href="https://github.com/DeepThink-Bench/Deepthinking"><b>[🐱 GitHub]</b></a>
  <br>
</p>
<p align="center">
This is a repository for building an unified framework and benchmark for DeepThink of LLMs.

# 💡 Abstract
Recent advancements in large language models (LLMs) have led to the development of large reasoning models (LRMs), which incorporate intermediate deep thinking to guide decision-making. These LRMs have demonstrated promising results in a range of domains, including commonsense reasoning, mathematics, and code generation. However, the precise role of deep thinking in improving model performance remains underexplored, and no universally accepted framework exists to evaluate its impact. To address this gap, we introduce \textsc{TGBench}, a comprehensive benchmarking framework designed to evaluate the effects of deep thinking on instruction-based LLMs. Our experiments reveal three key findings: 1) incorporating deep thinking from LRMs significantly enhances the performance of instruction-based LLMs, particularly in tasks that require multi-step reasoning; 2) deep thinking improves both accuracy and efficiency, though the extent of improvement varies depending on the task; and 3) we propose three distinct rankings (i.e., ranking single LLMs, ranking single LRMs, and ranking combined LLMs), providing a holistic view of deep thinking. These contributions highlight the potential of integrating deep thinking to advance instruction-based LLM capabilities, and we advocate for further research on optimizing deep thinking integration to enhance model scalability, robustness, and real-world applicability across diverse tasks.
<p align="center">
    <img src="https://github.com/DeepThink-Bench/Deepthinking/blob/main/utils/overview%20.png" width="1000">
        <br>
    <em>An overview of the CorrectBench framework.</em>
</p>


# 📃Overview of this project
- **./config:** The config file of the models (`./config/model_config`).

- **./dataset:** Public datasets we used. 

- **./method:** The code files we use to make the dataset include Base.py、Think.py、Fewshot.py.
  
- **./model:** Definition of loading API models and local models.

- **./result_base:** The DeepThinking TGBench-base dataset we created.

- **./results_fewshot:** The DeepThinking TGBench-fewshot dataset we created.

- **./results_think:** The DeepThinking TGBench-think dataset we created.

- **./utils:** Other assistive utilities.

# 🚀Preparation 
- **Environment settings:** `pip install -r ./requirement.txt`

- **Model settings:**   
Using API model of GPT series and Claude (see the [model list](https://api.keya.pw/pricing)), refer to the config file `./config/model_config/api_gpt_config.json`, set `"YOUR_API_KEY"` to your own API keys and `"model_method"` to `"api"`.   
Using other open-source API models from DeepInfra (see the [model list](https://deepinfra.com/models)), refer to the config file `./config/model_config/api_llama_config.json`, set `"YOUR_API_KEY"` to your own API keys and `"model_method"` to `"api"`.

# 😋Usage 
- **Usage Demo:** `./method/Base.py` provides a demo for using to solve dataset. Here is a demo for fast usage and other similar methods:   
```
parser = argparse.ArgumentParser(description="Prompt-based Testing and Saving Script for Multiple Tasks")
parser.add_argument('--model_config', type=str, default="/deep-thinking/config/model_config/api_deepseek-R1-zero_config.json",
                    help='Path to the model configuration file.')
parser.add_argument('--task_config_dir', type=str, default="/deep-thinking/dataset_3",
                    help='Path to the directory containing task configuration files.')
parser.add_argument('--method', type=str, default='Base',
                    help='Method name to use.')
parser.add_argument('--prompting_style', type=str, default='zero-shot-cot',
                    choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
                    help='Prompting style to use.')
parser.add_argument('--correct_iteration', type=int, default=1,
                    help='Number of correction iterations.')
args = parser.parse_args()

test_and_save(args)

```

You can use `--model_config` to specify the model to use, and use `--task_config_dir` to specify the test data set, which is stored in the `results/{args.method}/{task.task_name}/'
results_file = f'{results_path}/{model.name}_results.json` folder by default. Here is an example:
```sh
python Base.py \
    --model_config <Your Model Path> \
    --task_config_dir <Your Tasks Path> \
    --method rci\
    --prompting_style zero-shot-cot \
    --correct_iteration 1
```
Here's an example of what a JSON line in a **generation result file** might look like:
```json lines
{
"ACC": "ACC",
"empty_answers": "empty_answer_count",
"results": "final_results"
}
```
