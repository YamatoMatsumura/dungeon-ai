# Dungeon AI

## Description
This project features an autonomous agent that uses computer vision to identify game states and make real time decisions to successfully complete a dungeon.

## Motivation
This project grew out of a curiosity for computer vision and the repetitive nature of grinding dungeons in the game *Realm of the Mad God*. The goal was to build an autonomous agent capable of playing through the entirety of the dungeon using only visual inputs, while also gaining hands on experience applying computer vision techniques to a real time setting.

## Installation

```bash
# Clone the repository
git clone https://github.com/YamatoMatsumura/dungeon-ai.git
cd dungeon-ai

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

> Note: The player must be in the Pirate Cave dungeon as well as have the screen visible within two seconds after running main for the agent to function correctly.

## How It Works
1. Captures frames from the game window in real time
2. Applies computer vision techniques to understand the current game state
3. Selects and executes actions based on the detected state of the game

## Example Result

## Computer Vision Approach
