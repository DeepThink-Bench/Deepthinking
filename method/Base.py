import re
import os
import json
from tqdm import tqdm
import argparse
import glob
import sys
import torch
import time  # Importing the time module
sys.path.append('/deep-thinking')

# Ensure CUDA_VISIBLE_DEVICES is set
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

class Prompt_based:
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1):
        self.model = model
        self.task = task
        self.prompting_style = prompting_style
        self.correct_iteration = correct_iteration
        self.initial_prompt = self.get_initial_prompt()
        self.critique_prompt = 'Review your previous answer and find problems with your answer.\n\n'
        self.improve_prompt = (
            'Based on the problems you found, improve your answer. '
            'in the form \\boxed{answer}.\n\n'
        )

    def get_initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            return (
                "in the form \\boxed{answer}, at the end of your response.\n\nA:"
            )
        elif self.prompting_style == 'few-shot-cot':
            return "A:\n"  # TODO: add the few-shot prompt file
        elif self.prompting_style == 'zero-shot':
            return (
                "Your final answer in the form \\boxed{answer}, "
                "at the end of your response.\nA:\n"
            )
        else:
            print("WARNING: The prompting style is not given. Use zero-shot-cot as default.")
            return (
                "Let's think step by step.  "
                "in the form \\boxed{answer}, at the end of your response.\nA:\n"
            )

    # def get_answer(self, output):
    #     """
    #     Extracts the answer from the model's output. It looks for the pattern \boxed{answer}.
    #     """

    #     answer = re.findall(r'\\boxed{(.+?)}', output)
    #     if answer:
    #         try:
    #             return int(answer[0])
    #         except ValueError:
    #             try:
    #                 return float(answer[0])
    #             except ValueError:
    #                 return answer[0]  # Return as string if not a number
    #     else:
    #         return None  # Indicate no answer found


    def get_answer(self, output):
        """
        Extracts the answer from the model's output. It looks for the pattern \boxed{answer}.
        """

        # **确保 output 是字符串**
        if not isinstance(output, str):
            print(f"⚠️ Warning: output is not a string (type={type(output)}), converting to string.")
            output = str(output)  # 转换为字符串，防止 TypeError

        # **使用正则匹配 \boxed{...}**
        answer = re.findall(r'\\boxed{(.+?)}', output)

        # **如果找到答案**
        if answer:
            extracted_answer = answer[0].strip()  # 去掉前后空格
            try:
                return int(extracted_answer)  # 尝试转换为整数
            except ValueError:
                try:
                    return float(extracted_answer)  # 尝试转换为浮点数
                except ValueError:
                    return extracted_answer  # 返回原字符串

        # **未找到答案**
        print("⚠️ No answer found in output.")
        return None

    def __call__(self, question, answer):
        
        prompt_based = "Please solve the question above, then store the final answer in \\boxed{answer}."
        initial_input = 'Q: ' + question + '\n\n' + prompt_based
        output = self.model.query(initial_input)
        
        final_answer = self.get_answer(output)
                
        record = {}
        record['question'] = initial_input
        record['output'] = output
        
        record['final_answer'] = final_answer
        record['correct_answer'] = answer
        print("-----------------------------------------")
        print(f"final_answer:{final_answer}")
        print(f"correct_answer:{answer}")
        print("-----------------------------------------")
        
        if final_answer is None:
            record['correct'] = False
            record['error'] = 'No boxed answer found'
        if str(final_answer) == str(answer):
            record['correct'] = True
        else:
            record['correct'] = False

        
        if not record.get('correct', False):
                record['error'] = 'Final answer and answer do not match'
        return record


def test_and_save(args):
    from utils.process_config import open_config
    from model import create_model

    # Record the start time
    start_time = time.time()  # Start the timer

    # Load model configuration and create model
    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    # Get all task configuration files in the task_config_dir
    task_config_files = glob.glob(os.path.join(args.task_config_dir, '*.json'))

    if not task_config_files:
        print(f"No task configuration files found in {args.task_config_dir}.")
        return

    for task_config_path in task_config_files:
        task_name = os.path.splitext(os.path.basename(task_config_path))[0]
        print(f"\nProcessing Task: {task_name}")

        # Load task configuration and data
        with open(task_config_path, 'r') as f:
            task_data = json.load(f)

        # Ensure that the task data contains questions and answers
        if not isinstance(task_data, list) or not all(
            ("Question" in item or "Question'" in item) and "Answer" in item for item in task_data
        ):
            print(f"Skipping {task_config_path} because it does not contain 'Question' and 'Answer'.")
            continue

        # Normalize the keys to handle extra quotes or apostrophes
        for item in task_data:
            if "Question'" in item:
                item["Question"] = item.pop("Question'")  # Fix key if there is an extra apostrophe
            if "Question" not in item:
                print(f"Missing 'Question' key in task data for {task_config_path}")
                continue

        # Now, we can safely access 'Question' and 'Answer'
        questions = [item["Question"] for item in task_data]
        correct_answers = [item["Answer"] for item in task_data]

        # Initialize the correction method
        method = Prompt_based(model, task=None, prompting_style=args.prompting_style, correct_iteration=args.correct_iteration)

        # Create a directory to store the results for this task
        results_path = f'/deep-thinking/result_no_think/{args.method}/{task_name}/'
        results_file = f'{results_path}/{model.name}_results.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        print(f"Making a new file {results_file} to save the result.")

        final_results = []
        correct_number = 0
        total_number = len(questions)
        empty_answer_count = 0

        for i in tqdm(range(total_number), desc=f"Processing {task_name} questions"):
            question = questions[i]
            answer = correct_answers[i]
            record = method(question, answer)
            final_results.append(record)
            if record.get('correct', False):
                correct_number += 1
            if record.get('final_answer') is None:
                empty_answer_count += 1

            # Calculate the accuracy and update the results in real-time
            ACC = correct_number / (i + 1 - empty_answer_count) if (i + 1 - empty_answer_count) > 0 else 0
            results_dict = {
                "ACC": ACC,
                "empty_answers": empty_answer_count,
                "results": final_results
            }

            # Save the results to the file after processing each question
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=4)

        # Record the end time
        end_time = time.time()  # End the timer
        total_time = end_time - start_time  # Calculate total time

        print(f"Method: {args.method}\nTask: {task_name}\nModel: {model.name}\nFinal Accuracy: {ACC:.2f}")
        print(f"Number of questions with empty answers: {empty_answer_count}")
        print(f"Total runtime: {total_time:.2f} seconds")  # Print the total runtime

        # Update results_dict with runtime info
        results_dict["time"] = total_time

        # Save the updated results including runtime information
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)

        print(f"Results saved to {results_file}")


def main():
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


if __name__ == "__main__":
    main()
