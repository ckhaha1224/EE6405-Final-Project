from flask import Flask, render_template, request, jsonify
import time
from vllm import LLM, SamplingParams

app = Flask(__name__)

# 使用 VLLM 加载模型
print("Loading model with VLLM...")
model = LLM(
    model="./model_Emollama-chat-7b/",
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="float16"  # 使用 float16 提高效率
)
print("Model loaded successfully!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()

    # 获取请求数据
    data = request.json
    task_type = data.get('task_type', '')
    text = data.get('text', '')
    emotion = data.get('emotion', '')

    # 定义任务描述
    task_descriptions = {
        'emotion_intensity': f"Assign a numerical value between 0 (least {emotion}) and 1 (most {emotion}) to represent the intensity of emotion {emotion} expressed in the text.",
        'sentiment_strength': "Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).",
        'sentiment_classification': "Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred",
        'emotion_classification': "Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust)."
    }

    # 获取相应的任务描述
    task = task_descriptions.get(task_type, '')

    # 根据任务类型创建提示
    if task_type == 'emotion_intensity':
        prompt = f'''
Human:
Task: {task}
Text: {text}
Emotion: {emotion}
Intensity Score:

A:

'''
    elif task_type == 'sentiment_strength':
        prompt = f'''
Human:
Task: {task}
Text: {text}
Intensity Score:

A:

'''
    elif task_type == 'sentiment_classification':
        prompt = f'''
Human:
Task: {task}
Text: {text}
Intensity Class:

A:

'''
    elif task_type == 'emotion_classification':
        prompt = f'''
Human:
Task: {task}
Text: {text}
This text contains emotions:

A:

'''

    # 使用 VLLM 进行推理
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9
    )
    
    outputs = model.generate(prompt, sampling_params)
    response = outputs[0].outputs[0].text.strip()

    # 提取模型回答部分
    result = response.split("A:")[-1].strip()

    # 计算处理时间
    processing_time = time.time() - start_time

    return jsonify({
        'result': result,
        'processing_time': f"{processing_time:.2f} seconds"
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)