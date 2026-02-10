"""
OSWorld projection functions for converting text actions to environment actions.
Based on the BrowserGym projection pattern but directly uses parsed_pyautogui_code.
"""

import re
import json
from typing import List, Tuple, Optional
from agent_system.environments.env_package.OSWorld.ui_tars_action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code
from agent_system.environments.env_package.OSWorld.qwen3vl_action_parser import parse_qwen3vl_response


def osworld_projection(
    text_actions: List[str],
    model_type: str = "qwen25vl",
    screen_sizes: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[str], List[bool]]:
    """
    Project text actions to pyautogui code for OSWorld environment.
    
    Args:
        text_actions: List of model output strings
        model_type: Model type for parsing ("qwen25vl", "qwen3vl", etc.)
    
    Returns:
        Tuple of (parsed_actions_code, valids)
    """
    if screen_sizes is None:
        raise ValueError("osworld_projection requires screen_sizes (per-env real resolution); got None")
    valids = [0] * len(text_actions)
    parsed_actions_code = []

    for i in range(len(text_actions)):
        response = text_actions[i]  # keep the original string
        original_image_width, original_image_height = screen_sizes[i]
        original_image_width = int(original_image_width)
        original_image_height = int(original_image_height)
        if original_image_width <= 0 or original_image_height <= 0:
            raise ValueError(f"Invalid screen size: {original_image_width}x{original_image_height}")

        try:
            # Choose parser based on model type
            if model_type == "qwen3vl":
                # Use Qwen3-VL specific parser (same logic as qwen3vl_agent.py)
                # Directly parse to pyautogui code without intermediate conversion
                low_level_instruction, parsed_pyautogui_code = parse_qwen3vl_response(
                    response=response,
                    original_width=original_image_width,
                    original_height=original_image_height,
                    processed_width=None,
                    processed_height=None,
                    coordinate_type="relative"  # Qwen3-VL uses relative coordinates (0-999)
                )
            else:
                # Use UI-TARS parser (default for qwen25vl)
                parsed_dict = parse_action_to_structure_output(
                    response,
                    factor=1000,
                    origin_resized_height=original_image_height,
                    origin_resized_width=original_image_width,
                    model_type=model_type
                )
                # Convert to pyautogui code
                parsed_pyautogui_code = parsing_response_to_pyautogui_code(
                    responses=parsed_dict,
                    image_height=original_image_height,
                    image_width=original_image_width
                )
            
            parsed_actions_code.append(parsed_pyautogui_code)
            valids[i] = 1

        except Exception as e:
            valids[i] = 0
            parsed_actions_code.append(response)  # Use original response as fallback
    
    return parsed_actions_code, valids