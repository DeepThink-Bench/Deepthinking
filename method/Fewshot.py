import re
import os
import json
from tqdm import tqdm
import argparse
import sys
import torch
import time
sys.path.append('/deep-thinking')

# Ensure CUDA_VISIBLE_DEVICES is set
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
 


def extract_outputs(json_file_path):
    """
    从给定的 JSON 文件中提取所有 "output" 值并返回一个列表，
    同时计算这些 output 的总 token 数（使用空格简单切分）。
    
    参数:
    - json_file_path (str): JSON 文件的路径。

    返回:
    - Tuple[List[str], int]: (outputs 列表, outputs 总 token 数).
    """
    outputs = []
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        

        if 'results' in data and isinstance(data['results'], list):
            for idx, result in enumerate(data['results']):
                output = result.get('output')
                if output:
                    outputs.append(output)
                else:
                    print(f"警告: 'output' 在结果索引 {idx} 中不存在。")
        else:
            print("错误: JSON 中缺少 'results' 键或其格式不正确。")
    
    total_token_count = sum(len(o.split()) for o in outputs)
    
    return outputs, total_token_count


class Prompt_based:
    def __init__(self, model, task=None, prompting_style='zero-shot-cot', correct_iteration=1, thinking_outputs=None):
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

        self.thinking_outputs = thinking_outputs if thinking_outputs else []

    def get_initial_prompt(self):
        if self.prompting_style == 'zero-shot-cot':
            return "in the form \\boxed{answer}, at the end of your response.\n\nA:"
        elif self.prompting_style == 'few-shot-cot':
            return "A:\n"  # TODO: 
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
    

    def get_answer(self, output):
        """
        Extracts the answer from the model's output. It looks for the pattern \boxed{answer}.
        """
        answer = re.findall(r'\\boxed{(.+?)}', output)
        if answer:

            try:
                return int(answer[0])
            except ValueError:
                try:
                    return float(answer[0])
                except ValueError:
                    return answer[0]  
        else:
            return None  

    def __call__(self, question, answer, i, task_name):

        thinking = ''.join(self.thinking_outputs[i]) if self.thinking_outputs else ''
        
        prompt_based = "Please provide the final answer and store it in \\boxed{answer}."
        few_shot = {'hellswag':"""
                                In response to the ##Question, please output according to the following response format:
                                # Example 1:

                                ## Question:
                                [header] How to use a dutch oven [title] Bake with your dutch oven. [step] You can cook bread, pizzas, cakes, and other desserts in your dutch oven by placing hot coals on top of the lid and underneath the dutch oven full of food. To bake, you should place more coals on top of the lid than underneath the dutch oven. 0. This will encourage the food to expand as it bakes. [title] Cook with the dutch oven on a low flame. Please choose the most reasonable scenario outcome from the multiple options above,1. [substeps] Always check the temperature of your dutch oven using an oven thermometer. Don't bake the dutch oven at a temperature higher than that 200 degrees fahrenheit (74 degrees celsius). Please choose the most reasonable scenario outcome from the multiple options above,2. This will prevent food on the bottom from burning. [substeps] Consider your dutch oven's diameter. Please choose the most reasonable scenario outcome from the multiple options above,3. Alternatively, you can use a gas or electric dutch oven. [substeps] If you have any special cooking ingredients in your dutch oven, you can use them to pan over or preheat it for the next step in the recipe. Please choose the most reasonable scenario outcome from the multiple options above,

                                ## Response:
                                ### Deep Thinking:
                                To solve this, we need to focus on the key steps to properly bake with a dutch oven. Baking with coals on top will help with food expansion, and maintaining the oven at a low temperature will prevent burning.

                                ### Answer:
                                The final answer in \boxed{200 degrees fahrenheit}. 


                                Example 2:

                                ## Question:
                                [header] How to format a legal pleading [title] Find a pleading form. [step] Some courts have " check the box " or " fill in the blank " pleading forms, which make the process easy. First, see if your court has form pleadings. 0. [substeps] If your court does not have " check the box " or " fill in the blank " pleading forms, then you can download the pleading form from a court website. There are also " trial pleading forms " which often contain numbered paragraphs. Please choose the most reasonable scenario outcome from the multiple options above,1. [substeps] Often, these should be listed on the website for the court. If you can't find any, call and ask the clerk of court. Please choose the most reasonable scenario outcome from the multiple options above,2. [substeps] In common pleading forms you'll find: pleading " guilty " or " perjury. " your pleading will show you how you were feeling after a criminal conviction, and it should be formatted as " guilty " or " perjury ". Please choose the most reasonable scenario outcome from the multiple options above,3. Make sure you've, but sometimes not, filled out forms properly when you mail them. [substeps] If you are seeking a settlement in excess of $200 , 000, then you must include a separate " check box " in your pleading. Please choose the most reasonable scenario outcome from the multiple options above,

                                ## Response:
                                ### Deep Thinking:
                                Here, we are analyzing the necessary steps for finding the correct pleading form. If you can’t find a " check the box " or " fill in the blank " form, download it from the court website or contact the clerk.

                                ### Answer:
                                The final answer in \boxed{download the pleading form}.
                                """,
                    'math':"""
                                In response to the ##Question, please output according to the following response format:
                                # Example 1:
                                ## Question:
                                The Bank of Springfield's Super High Yield savings account compounds annually at a rate of one percent. If Lisa invests 1000 dollars in one of these accounts, then how much interest will she earn after five years? (Give your answer to the nearest dollar.)

                                ## Response:
                                ### Deep Thinking:
                                Use the formula for compound interest: A = P(1 + r/n)^(nt). In this case, P = 1000, r = 0.01, t = 5 years, and n = 1 (compounding annually).

                                ### Answer:
                                The final answer in \boxed{51}. 

                                # Example 2:
                                ## Question:
                                Find the sum of $327_8$ and $73_8$ in base $8$.

                                ## Response:
                                ### Deep Thinking:
                                To solve, first convert both numbers from base 8 to base 10, then add them, and convert the sum back to base 8.

                                ### Answer:
                                The final answer in \boxed{502_8}. 
                                """
,
                    'mbpp':"""
                                In response to the ##Question, please output according to the following response format:
                                # Example 1:
                                ## Question:
                                Write a python function to accept the strings which contains all vowels.

                                ## Response:
                                ### Deep Thinking:
                                We need to check if a string contains all vowels ('a', 'e', 'i', 'o', 'u'). The function should iterate through the string to ensure each vowel appears.

                                ### Answer:
                                The final answer in \boxed{def contains_all_vowels(s): return all(vowel in s for vowel in 'aeiou')}. 

                                # Example 2:
                                ## Question:
                                Write a function to find minimum of two numbers.

                                ## Response:
                                ### Deep Thinking:
                                We need to compare two numbers and return the smaller one. This can be done using a simple comparison.

                                ### Answer:
                                The final answer in \boxed{def min_of_two(a, b): return a if a < b else b}. 
                                """
,
                    'imdb':"""
                                In response to the ##Question, please output according to the following response format:
                                # Example 1:

                                ## Question:
                                Peter Weir's first international success, THE LAST WAVE is a mainly effective chiller with a fascinating back story based on Aboriginal myth. Richard Chamberlain gives a good performance as a defense lawyer whose life becomes increasingly unmoored from reality as he delves deeper into a murder case involving Aboriginal tribal rivalries. David Gulpilil plays one of the suspects, who does his best to guide Chamberlin thru the realm of 'Dreamtime', an alternate reality/timeline central to native Australian history and tribal custom. Heavy on atmosphere, deliberately ambiguous in plotting, the film builds to an unsettling finale which is somewhat diminished by poor effects, probably due to budgetary limitations. Nevertheless an intriguing film whose overall impression of mystery and dread lurking just below the surface of what we perceive as 'reality' will stay with you. If you think this sentence is positive, the answer is 1, and the negative answer is 0

                                ## Response:
                                ### Deep Thinking:
                                The review describes the film positively, mentioning a fascinating backstory and good performances despite some technical limitations.

                                ### Answer:
                                The final answer in \boxed{1}. 


                                # Example 2:

                                ## Question:
                                When this movie first came out back in 1984, Prince was one of the hottest acts around. Everyone wanted to see this movie, which was not much more than an extended music video. The acting was pretty bad, but what can you expect from musicians acting on the big screen for the first time? Despite that, it was still a very entertaining film! Morris Day and Jerome Benton provide some all-time classic comedy, especially their rendition of "The Password", which will make you think of Abbott & Costello doing their "who's on first" baseball routine.<br /><br />Appolina (who went by a single name then) provided some beautiful breasts, so you had the brief nudity covered. Plus, she is very attractive. And of course, the soundtrack of the album is one of the best Prince ever recorded. Prince later on had a fallout with Warner Bros. and changed his name, but at this particular time in his career, he was at the top of his game.<br /><br />This movie doesn't rank in the all-time great category, but it is pretty entertaining. If you think this sentence is positive, the answer is 1, and the negative answer is 0

                                ## Response:
                                ### Deep Thinking:
                                The review mentions both positive aspects like entertainment and the soundtrack, but also acknowledges weaknesses in acting.

                                ### Answer:
                                The final answer in \boxed{1}. 
                                """, 
                    'drop':"""
                                In response to the ##Question, please output according to the following response format:
                                # Example 1:

                                ## Question:
                                Colt McCoy completed 83.3 percent of his passes, (25-30 for 299 yds) the most in Redskins history by a quarterback with at least 30 attempts. Murray extended his 100-yard game rushing streak to eight games with 19 carries for 141 yards vs Washington. The game was the 107th meeting in 54 years between the Dallas Cowboys and the Washington Redskins. After the loss to Washington, the Cowboys now have a total of 64 wins to 41 losses and 2 ties in 54 years. Including the loss to the Redskins on Monday Night, the Cowboys have played in a total of 75 Monday Night games and have an overall record of 43-32. How many yards did Murray get?

                                ## Response:
                                ### Deep Thinking:
                                To solve the question, we need to extract the information provided about Murray’s performance. It is clearly stated that Murray rushed for 141 yards in the game against Washington. This is the key data point to answer the question.

                                ### Answer:
                                The final answer in \\boxed{141}. 


                                # Example 2:

                                ## Question:
                                Coming off their home win over the Ravens, the Chargers flew to Arrowhead Stadium for a Week 13 AFC West rematch with the Kansas City Chiefs. In the first quarter, San Diego trailed early as Chiefs kicker John Carney managed to get a 38-yard field goal. Afterwards, the Chargers got on the board with kicker Nate Kaeding nailing a 25-yard field goal. In the second quarter, Kansas City regained the lead as QB Damon Huard completed a 2-yard TD pass to DE Jared Allen. Afterwards, San Diego tied the game again as QB Philip Rivers completed a 38-yard TD pass to WR Vincent Jackson. In the second quarter, the Chargers pulled away as RB LaDainian Tomlinson got a 31-yard TD run in the third quarter and a 28-yard TD run in the fourth quarter. His two rushing touchdowns helped him surpass Walter Payton for third place on the NFL's all-time rushing touchdowns list. The game also gave him his 3rd-straight 100-yard game against the Chiefs. Also, CB Antonio Cromartie recorded 2 INT to bring his league-leading total to 8. How many yards longer was LaDainian Tomlinson's first touchdown compared to his second?

                                ## Response:
                                ### Deep Thinking:
                                To solve the question, we need to compare the lengths of LaDainian Tomlinson’s two touchdowns. The first touchdown is a 31-yard run and the second touchdown is a 28-yard run. The difference in length is calculated as 31 - 28 yards.

                                ### Answer:
                                The final answer in \\boxed{3}.
"""}  

        initial_input = (
            'Given the question statement:' + question + '\n\n'
            + 'Use following thought to solve it:' + thinking + '\n\n'
            + 'Examples: ' + few_shot.get(task_name, '') + '\n\n' 
            + prompt_based
        )

        output = self.model.query(initial_input)
        
        final_answer = self.get_answer(output)
                
        record = {}
        record['question'] = initial_input
        record['output'] = output
        record['final_answer'] = final_answer
        record['correct_answer'] = answer
        
        print("-----------------------------------------")
        print(f"final_answer: {final_answer}")
        print(f"correct_answer: {answer}")
        print("-----------------------------------------")
        
        if final_answer is None:
            record['correct'] = False
            record['error'] = 'No boxed answer found'
        elif str(final_answer) == str(answer):
            record['correct'] = True
        else:
            record['correct'] = False

        if not record.get('correct', False):
            record['error'] = 'Final answer and answer do not match'
        return record


def test_and_save(args):
    """
    1. 加载模型
    2. 加载 single task 配置文件 (从 --task_config_file，而不是目录)
    3. 如果指定 --extra_json，则提取其中的 output 并统计 token 数，作为 thinking 传给 Prompt_based
    4. 逐条处理问题并保存结果
    """
    from utils.process_config import open_config
    from model import create_model

    start_time = time.time()

    model_config = open_config(config_path=args.model_config)
    model = create_model(model_config)

    if not os.path.isfile(args.task_config_file):
        print(f"[Error] {args.task_config_file} is not a valid file.")
        return

    with open(args.task_config_file, 'r') as f:
        task_data = json.load(f)

    task_name = os.path.splitext(os.path.basename(args.task_config_file))[0]
    print(f"\nProcessing Task: {task_name}")

    thinking_outputs = []
    thinking_token_count = 0
    if args.extra_json:
        if os.path.isfile(args.extra_json):
            thinking_outputs, thinking_token_count = extract_outputs(args.extra_json)
        else:
            print(f"[Warning] {args.extra_json} is not a valid file. Skip extracting outputs.")

    if not isinstance(task_data, list) or not all(
        ("Question" in item or "Question'" in item) and "Answer" in item for item in task_data
    ):
        print(f"Skipping {args.task_config_file} because it does not contain 'Question' and 'Answer'.")
        return

    for item in task_data:
        if "Question'" in item:
            item["Question"] = item.pop("Question'")

    questions = [item["Question"] for item in task_data]
    correct_answers = [item["Answer"] for item in task_data]

    method = Prompt_based(
        model, 
        task=None, 
        prompting_style=args.prompting_style, 
        correct_iteration=args.correct_iteration,
        thinking_outputs=thinking_outputs
    )

    results_path = f'/deep-thinking/results_fewshot/{args.method}/{task_name}/'
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
        record = method(question, answer, i, task_name)  # Pass task_name here
        final_results.append(record)
        
        if record.get('correct', False):
            correct_number += 1
        if record.get('final_answer') is None:
            empty_answer_count += 1

    answered_count = (i + 1 - empty_answer_count)
    ACC = correct_number / answered_count if answered_count > 0 else 0

    results_dict = {
        "ACC": ACC,
        "thinking_token_count": thinking_token_count,  
        "empty_answers": empty_answer_count,
        "results": final_results
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Method: {args.method}")
    print(f"Task: {task_name}")
    print(f"Model: {model.name}")
    print(f"Final Accuracy: {ACC:.2f}")
    print(f"Thinking token count: {thinking_token_count}")
    print(f"Number of questions with empty answers: {empty_answer_count}")
    print(f"Total runtime: {total_time:.2f} seconds")

    results_dict["time"] = total_time
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Prompt-based Testing and Saving Script for a Single Task")
    parser.add_argument('--model_config', type=str, default="/deep-thinking/config/model_config/api_deepseek-R1_config.json",
                        help='Path to the model configuration file.')
    parser.add_argument('--task_config_file', type=str, required=True,
                        help='Path to the single task JSON file (contains questions & answers).')
    parser.add_argument('--method', type=str, default='gemini-2.0-flash-thinking-exp',
                        help='Method name to use.')
    parser.add_argument('--prompting_style', type=str, default='zero-shot-cot',
                        choices=['zero-shot-cot', 'few-shot-cot', 'zero-shot'],
                        help='Prompting style to use.')
    parser.add_argument('--correct_iteration', type=int, default=1,
                        help='Number of correction iterations.')

    parser.add_argument('--extra_json', type=str, default=None,
                        help='Path to the additional JSON file containing outputs to be used as thinking.')
    args = parser.parse_args()

    test_and_save(args)


if __name__ == "__main__":
    main()
