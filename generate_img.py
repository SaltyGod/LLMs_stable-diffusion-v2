from prompt import SYSTEM_PROMPT
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2
import random
from tqdm import tqdm
import json
import csv
import os
device = "cuda" 


def img_style(characters,styles):
    """
    随机选择img的主要角色以及绘画风格
    
    Args:
        characters (list): 主要角色列表
        styles (list): 绘画样式
    
    Returns:
        tuple: 返回元组。
    
    """
        
    character = random.choice(characters)
    style = random.choice(styles)
    
    return character,style


def keywords_to_str(file_path, character):
    with open(file_path, 'r', encoding='utf-8') as file:
        keywords = [line.strip() for line in file.readlines()]
    
    # 添加随机抽取的chatater
    keywords.append(character)
    
    # 打乱重排
    random.shuffle(keywords)
    
    # 拼接字符串
    result = ','.join(keywords)
    
    return result
    

def generate_text(model_dir,query):
    """
    根据给定的问题，使用预训练的生成式对话模型生成回答
    
    Args:
        model_dir (str): 模型路径
        query (str): 输入提问
    
    Returns:
        str: 解码后的回答
    
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=25
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print('场景生成完毕！')
    
    return response


def generate_img(model_id, response, character, i, output_dir):
    """
    基于LLMs生成的文字场景，绘制img
    
    Args:
        model_id (str): 模型路径。
        response (str): 响应内容，作为生成图像的文本输入。
        character (str): 角色名，用于生成图像的文件名。
        i (int): 图片编号，用于生成图像的文件名。
    
    Returns:
        None
    """
    pipe = pipeline(task=Tasks.text_to_image_synthesis,
                    model=model_id,
                    model_revision='v1.0.0')
    
    output = pipe({'text': response})
    file_path = os.path.join(output_dir, f'{character}_{i}.png')
    
    # 保存图像
    cv2.imwrite(file_path, output['output_imgs'][0])
    
    

def to_csv(sence_output,results):
    csv_columns = ["keywords", "character", "scene description"]

    with open(sence_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
        # 提取results中的数据并写入CSV文件
        for result in results:
            conversation = result["conversation"][0]
            writer.writerow(conversation)

    print("CSV file generated successfully.")


########################################### 执行函数 ###########################################
# 定义参数
file_path = './keyword.txt'
characters = ['cat','dog','cute sheep'] # 绘画主人公
styles = ['Anime', 'pixel', 'hand-painted','ink', 'impressionism', 'minimalism', 'watercolor painting','sand painting','Realism'] # 绘画风格
model_dir = './Qwen1.5-1.8B-Chat'
model_id = './AI-ModelScope/stable-diffusion-v2-1'
sence_output = './output/sence.csv'
output_dir = './output/img_output/'
os.makedirs(output_dir, exist_ok=True)

# 循环生图
results=[]
i = 0
for _ in tqdm(range(5), desc="Processing"): 
    i += 1
    character, style = img_style(characters, styles)
    result = keywords_to_str(file_path, character)
    query = SYSTEM_PROMPT.format(keywords=result, character=character)
    
    response = generate_text(model_dir, query)
    response = style + ',' + response  # 拼接绘画风格到response开头
    generate_img(model_id, response, character, i, output_dir)
    
    results.append(
    {
        "conversation": [
            {
                "keywords": result,
                "character": character,
                "scene description":response
            }
        ]
    }
    )
    
    to_csv(sence_output,results)
        

        
        

    
        
        
