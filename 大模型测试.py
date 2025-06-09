import matplotlib.pyplot as plt
import numpy as np
import logging
import ast
from openai import OpenAI

client = OpenAI(
    api_key="xai-YauNBWEwg7LsL4qCkoCTCKlyub5wqsYRNUrY1cV5oPUb99MTxw1yFf4zvkrcQMQSQH1OrQxRSGbGp6x0",
    base_url="https://api.x.ai/v1"
)

logging.basicConfig(level=logging.INFO)

# 地图解析函数
def map_to_grid(map_text):
    lines = map_text.strip().split("\n")
    grid = np.zeros((len(lines), len(lines[0])), dtype=int)
    for i, row in enumerate(lines):
        for j, ch in enumerate(row):
            if ch == "#":
                grid[i, j] = 1
            elif ch == "E":
                grid[i, j] = 4  # 目标E
            elif ch == "1":
                grid[i, j] = 2  # 智能体1
            elif ch == "2":
                grid[i, j] = 3  # 智能体2
            elif ch == "3":
                grid[i, j] = 5  # 智能体3
    return grid

def plot_map_with_path(map_text, path_coords):
    grid = map_to_grid(map_text)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="gray_r")
    if path_coords:
        ys, xs = zip(*path_coords)
        plt.plot(xs, ys, color="red", linewidth=2, marker="o", markersize=4)
    plt.title("迷宫路径")
    plt.axis("off")
    plt.show()

def map_to_prompt(map_text):
    return f"""
你是一个任务分配模块，需要给智能体1,2,3分配任务让三个智能体去围追堵截智能体E，
字符说明：
- `1`: 智能体1
- `2`: 智能体2
- `3`: 智能体3
- `#`: 墙，不可通行
- 空格: 可通行区域
- `E`: 智能体E（目标）
以下是一个20x20字符地图：
{map_text}

请根据地图，给智能体1,2,3分配任务，让它们在距离5个格子的范围内封堵智能体E可能逃跑的路径。
请返回以下内容：
1. 每个智能体路径坐标列表，例如：[(1, 1), (1, 2), ..., (19, 18)]，不允许出现穿过墙的情况
2. 智能体1,2,3分别在哪几个点才能对智能体E形成围堵（封堵智能体E可能逃跑的路径）
3. 推理过程简要说明

请将这些信息封装为一个 JSON 字典返回，
不要输出任何多余内容。
"""

def get_directions(path_coords):
    dirs = []
    for (y1, x1), (y2, x2) in zip(path_coords, path_coords[1:]):
        dy, dx = y2 - y1, x2 - x1
        if dx == 1: dirs.append("right")
        elif dx == -1: dirs.append("left")
        elif dy == 1: dirs.append("down")
        elif dy == -1: dirs.append("up")
    return dirs

def query_grok_map_with_info(map_text):
    prompt = map_to_prompt(map_text)
    try:
        response = client.chat.completions.create(
            model="grok-3-beta",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        print("\n📦 原始返回：\n", raw)
        data = eval(raw, {"__builtins__": None}, {})
        return data, {"status": "success", **data}
    except Exception as e:
        logging.error(f"Grok API 请求失败: {e}")
        return [], {"status": "failure", "reason": str(e)}

# 示例地图
maze_map = """
####################
#1       ####      #
#        ####      #
#                 ##
#   ####           #
#   ####           #
#                  #
#        #######   #
#        #######   #
#                  #
# 2  ####       E  #
#    ####          #
#                  #
#          ######  #
#          ######  #
#                  #
#    ####          #
#    ####          #
#             3    #
####################

"""

# 调用执行
path_data, info = query_grok_map_with_info(maze_map)
print(path_data)
print(info)

