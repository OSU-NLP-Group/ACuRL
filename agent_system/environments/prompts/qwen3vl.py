# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Qwen3-VL specific prompts for OSWorld
# This file replicates the exact prompt structure from qwen3vl_agent.py

import json

# NOTE: Keep prompts compact to reduce token usage in rollout/eval.

# Action description for the action parameter
ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """

def get_description_prompt(coordinate_type: str = "relative", processed_width: int = None, processed_height: int = None) -> str:
    """
    Generate description prompt with dynamic resolution based on coordinate type.
    Exactly matches qwen3vl_agent.py logic.
    """
    description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
            (
                f"* The screen's resolution is {processed_width}x{processed_height}."
                if coordinate_type == "absolute"
                else "* The screen's resolution is 1000x1000."
            ),
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    description_prompt = "\n".join(description_prompt_lines)
    return description_prompt

def get_qwen3vl_system_prompt(coordinate_type: str = "relative", processed_width: int = None, processed_height: int = None) -> str:
    description_prompt = get_description_prompt(coordinate_type, processed_width, processed_height)
    
    tools_def = {
        "type": "function", 
        "function": {
            "name_for_human": "computer_use", 
            "name": "computer_use", 
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": ACTION_DESCRIPTION_PROMPT,
                        "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", 
                                 "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], 
                        "type": "string"
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"}, 
                    "text": {"description": "Required only by `action=type`.", "type": "string"}, 
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"}, 
                    "pixels": {"description": "The amount of scrolling.", "type": "number"}, 
                    "time": {"description": "The seconds to wait.", "type": "number"}, 
                    "status": {
                        "description": "The status of the task.", 
                        "type": "string", 
                        "enum": ["success", "failure"]
                    }
                }, 
                "required": ["action"], 
                "type": "object"
            }, 
            "args_format": "Format the arguments as a JSON object."
        }
    }
    
    system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""
    
    return system_prompt

# Default system prompt for backward compatibility (uses relative coordinates with 1000x1000)
QWEN3VL_SYSTEM_PROMPT = get_qwen3vl_system_prompt(coordinate_type="relative")

def get_qwen3vl_instruction_prompt(
    instruction: str, previous_actions_str: str | None = None, use_password: bool = False
):
    """
    Get instruction prompt for Qwen3-VL.
    
    Args:
        instruction: The task instruction
        previous_actions_str: String of previous actions (formatted as "Step N: action")
        use_password: If True, appends password note to the instruction
    
    Returns:
        Formatted instruction prompt
    """
    # Keep this prompt intentionally compact; the system prompt already defines strict formatting.
    prev = previous_actions_str if previous_actions_str else "None"
    base_prompt = (
        "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions: {prev}"
    )
    
    # Add password note to instruction if enabled
    if use_password:
        base_prompt += "\n\nNote: The computer's password is 'osworld-public-evaluation', feel free to use it when you need sudo rights."
    
    return base_prompt

