import tensorflow as tf
import os, json, re, io, base64, threading
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)


from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import tensorflow_datasets as tfds
from PIL import Image
import imageio
from tqdm import tqdm

from openai import OpenAI

# ---------------------------
# ★★★ 全局配置
# ---------------------------
DATASET_DIR      = '/mnt/hwfile/OpenRobotLab/robot_data/cache/fractal20220817_data/0.1.0'
SAVE_DIR         = Path("data_pipeline/batch_run_25k")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH        = SAVE_DIR / "output.json"

NUM_THREADS      = 5
EP_PER_THREAD    = 5_000          # 每线程 5 000 条
SAVE_EVERY       = 5            # 收集满 500 条再写盘

MODEL_NAME       = "qwen2.5-vl-72b-instruct"
API_CONFIG       = dict(
    api_key="sk-01c070ce110e4e3a8af73cacdaef8ea5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

buffer_lock = threading.Lock()      # 保护 output_dict 和 pending_cnt
output_dict: Dict[str, dict] = {}   # 所有已完成 annotation
pending_cnt = 0                     # 距上次落盘已累积的条数

if JSON_PATH.exists():              # 允许断点续跑
    with JSON_PATH.open() as f:
        output_dict.update(json.load(f))

def build_prompt(instruction):
    prompt_text = '''Imagine a robot assistant operating in a laboratory or household environment. The robot is expected to follow diverse commands based on realistic tasks and human interactions. Your task is to:

1. Write a caption to describe the visual scene shown in the **first image** in English. You should NOT include the robot itself here.
2. Based on the given robot task description and the images, generate new user instructions and corresponding robot responses in English with QA pairs.

The new user instructions should align with the actions performed by robot in the images, and with the environment shown in the images. You are required to produce three categories of instructions:
1. **Command Rewriting (CR)**: Rephrase the task description using diverse language styles and vocabulary. You may refer to objects by their utility, color, shape, or other attributes, but ensure the attribute you use is unique to each object.
2. **Context Creation (CC)**: Generate detailed scenarios where the robot needs to perform the given instruction. The situation should involve realistic surroundings or tasks where this instruction would be necessary. You may also simulate a long-horizon task based on the context provided by the image. Your generated question should NOT include the answer itself.
3. **Scene-related Commonsense QA (QA)**: Generate some other QA pairs that are related to the scene, which can be answered based on the first image. Each question should focus on object attributes or spatial relationships. The answer should be concise and consistent among the three images.

For each instruction, provide a concise robot response that clearly(use simple words) communicates the next action the robot will take. **Do not chain multiple actions together using phrases like "and then."** If necessary, the response may include a brief explanation of the reasoning. Avoid repeating the instruction in the response.

**Response Format**: You MUST respond in JSON format. You should include "Description", "Caption", "CR", "CC", and "QA" in your response. You should create 1-3 entries for each of CR, CC, and QA.
**Example 1**: For the instruction "Close middle drawer":
Corresponding three images (omitted)
**Caption**: "A table with a Coke and chips on top, with its middle drawer open."
{
    "Caption": "A table with a Coke and chips on top, with its middle drawer open.", # you should Not include the robot itself in the caption
    "CR": [
        {
            "question": "Push the middle drawer closed.",
            "answer": "Ok, I will close it."
        },
        {
            "question": "Ensure the center drawer is closed.",
            "answer": "I will close the drawer."
        }
    ],
    "CC": [
        {
            "question": "I want you to take out the Coke from the middle drawer and closing it.",
            "answer": "The Coke is on the table, and the middle drawer is empty. So, I should close the middle drawer." # the last step in a long-horizon task
        },
        {
            "question": "Please push the middle drawer shut so we can clear the workspace.",
            "answer": "Okay, I will close the middle drawer."
        }
    ],
    "QA": [
        {
            "question": "What is in the middle drawer?",
            "answer": "The middle drawer is empty."
        },
        {
            "question": "How many Coke cans are on the table?",
            "answer": "One."
        }
    ]
}
**Example 2**: For the instruction "move the apple near the Coke":
Corresponding three images (omitted)
**Caption**: "A table with Coke, apple, and soap on it."
{
    "Description": "Pick up the apple and place it near the Coke",
    "Caption": "A table with Coke, apple, and soap on it.", # You should NOT include the robot itself in the caption
    "CR": [
        {
            "question": "Move the healthy food near the Coke.",
            "answer": "The healthy food refers to the apple, and I will move the apple to the Coke." # Do NOT directly repeat the question from user!
        },
        {
            "question": "Move the apple to the cylindrical-shaped object.",
            "answer": "Of course!"
        }
    ],
    "CC": [
        {
            "question": "Gather all objects near the Coke, except the soap.",
            "answer": "I will move the apple to the Coke."  # The response is concise and contains only one action. Do NOT chain multiple actions together!
        }
    ],
    "QA": [
        {
            "question": "I'm thirsty, what can I have?",
            "answer": "The Coke is on the table."
        },
        {
            "question": "What is the healthy food on the table?",
            "answer": "The apple."
        }
    ]
}
Your task description is "<placeholder>". 
Now give your response in JSON format. Your response MUST NOT include any comments.'''.replace('<placeholder>', instruction )
    return prompt_text

def build_messages(img_urls, prompt_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_urls[0],
                        "detail":"low"
                        }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_urls[1],
                        "detail":"low"
                        }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_urls[2],
                        "detail":"low"
                        }
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]
    return messages


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="webp")
    return f"data:image/webp;base64,{base64.b64encode(buf.getvalue()).decode()}"

def clean_json_string(raw: str) -> str:
    return re.sub(r"^```(?:json)?\n?|\n?```$", "", raw.strip())

def save_episode_gif(ep_id, video):
    idx = [0, len(video)//4, len(video)//2, 3*len(video)//4, len(video)-1]
    frames = [Image.fromarray(video[i].numpy()) for i in idx]
    imageio.mimsave(SAVE_DIR / f"{ep_id}.gif", frames, duration=1)

def flush_to_disk(local_buf: Dict[str, dict]):
    global pending_cnt
    with buffer_lock:
        output_dict.update(local_buf)
        pending_cnt += len(local_buf)
        if pending_cnt >= SAVE_EVERY:
            with JSON_PATH.open("w") as f:
                json.dump(output_dict, f, indent=2)
            pending_cnt = 0


def worker(thread_idx: int, start_ep: int, end_ep: int):
    client  = OpenAI(**API_CONFIG)
    builder = tfds.builder_from_directory(DATASET_DIR)

    local_buf: Dict[str, dict] = {}

    for ep_idx in tqdm(range(start_ep, end_ep),
                       desc=f"Thread-{thread_idx}",
                       position=thread_idx):
        try:
            ds = builder.as_dataset(split=f"train[{ep_idx}:{ep_idx+1}]")
            episode = next(iter(ds))
            ep_id   = episode['episode_metadata']['episode_id'].numpy().decode()

            if ep_id in output_dict or ep_id in local_buf:
                continue

            video = [s["observation"]["image"] for s in episode["steps"]]
            instruction = [step["observation"]["natural_language_instruction"].numpy().decode() for step in episode["steps"]][0]

            imgs = [Image.fromarray(video[i].numpy()) for i in (0, len(video)//2, -1)]
            img_urls = [image_to_data_url(im) for im in imgs]

            prompt_text = build_prompt(instruction)
            messages = build_messages(img_urls, prompt_text)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False
            )
            raw = response.choices[0].message.content
            parsed = json.loads(clean_json_string(raw))

            local_buf[ep_id] = {
                "index": ep_idx,
                "annotation": parsed,
                "instruction": instruction,
            }

            if len(local_buf) >= SAVE_EVERY:
                flush_to_disk(local_buf)
                local_buf.clear()

        except Exception as exc:
            print(f"[Thread-{thread_idx}] ep {ep_idx} error: {exc}")

    if local_buf:
        flush_to_disk(local_buf)


def main():
    total_needed = NUM_THREADS * EP_PER_THREAD
    ranges = [
        (i * EP_PER_THREAD, (i + 1) * EP_PER_THREAD)
        for i in range(NUM_THREADS)
    ]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        futures = [
            pool.submit(worker, i, start, end)
            for i, (start, end) in enumerate(ranges)
        ]
        for f in as_completed(futures):
            f.result()

    with JSON_PATH.open("w") as f:
        json.dump(output_dict, f, indent=2)
    print("✔ All done. Total episodes:", len(output_dict))

# ---------------------------
if __name__ == "__main__":
    main()
