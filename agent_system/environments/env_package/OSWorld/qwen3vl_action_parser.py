import re
import json
from typing import List, Tuple, Dict


def parse_qwen3vl_response(
    response: str,
    original_width: int = 1920,
    original_height: int = 1080,
    processed_width: int = None,
    processed_height: int = None,
    coordinate_type: str = "relative"
) -> Tuple[str, List[str]]:
    """
    Parse Qwen3-VL LLM response and convert it to low level action and pyautogui code.
    
    Args:
        response: Raw model output text containing Action description and <tool_call> JSON
        original_width: Original screen width
        original_height: Original screen height  
        processed_width: Processed image width sent to model
        processed_height: Processed image height sent to model
        coordinate_type: "relative" (0-999 grid) or "absolute" (pixel coordinates)
    
    Returns:
        Tuple of (low_level_instruction, pyautogui_code_string)
    """
    low_level_instruction = ""
    pyautogui_code: List[str] = []

    if response is None or not response.strip():
        return low_level_instruction, pyautogui_code

    def adjust_coordinates(x: float, y: float) -> Tuple[int, int]:
        """Adjust coordinates from model output to original screen coordinates"""
        if not (original_width and original_height):
            return int(x), int(y)
        if coordinate_type == "absolute":
            # scale from processed pixels to original
            if processed_width and processed_height:
                x_scale = original_width / processed_width
                y_scale = original_height / processed_height
                return int(x * x_scale), int(y * y_scale)
            return int(x), int(y)
        # relative: scale from 0..999 grid
        x_scale = original_width / 999
        y_scale = original_height / 999
        return int(x * x_scale), int(y * y_scale)

    def process_tool_call(json_str: str) -> None:
        """Process a single tool call JSON and generate pyautogui code"""
        try:
            tool_call = json.loads(json_str)
            if tool_call.get("name") == "computer_use":
                args = tool_call["arguments"]
                action = args["action"]

                if action == "left_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(x, y)
                        pyautogui_code.append(f"pyautogui.click({adj_x}, {adj_y})")
                    else:
                        pyautogui_code.append("pyautogui.click()")

                elif action == "right_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(x, y)
                        pyautogui_code.append(
                            f"pyautogui.rightClick({adj_x}, {adj_y})"
                        )
                    else:
                        pyautogui_code.append("pyautogui.rightClick()")

                elif action == "middle_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(x, y)
                        pyautogui_code.append(
                            f"pyautogui.middleClick({adj_x}, {adj_y})"
                        )
                    else:
                        pyautogui_code.append("pyautogui.middleClick()")

                elif action == "double_click":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(x, y)
                        pyautogui_code.append(
                            f"pyautogui.doubleClick({adj_x}, {adj_y})"
                        )
                    else:
                        pyautogui_code.append("pyautogui.doubleClick()")

                elif action == "type":
                    text = args.get("text", "")
                    # Escape single quotes for Python code
                    text = text.replace("'", "\\'")
                    pyautogui_code.append(f"pyautogui.typewrite('{text}')")

                elif action == "key":
                    keys = args.get("keys", [])
                    if isinstance(keys, list):
                        cleaned_keys = []
                        for key in keys:
                            if isinstance(key, str):
                                # Clean up any malformed key strings
                                if key.startswith("keys=["):
                                    key = key[6:]
                                if key.endswith("]"):
                                    key = key[:-1]
                                if key.startswith("['") or key.startswith('["'):
                                    key = key[2:] if len(key) > 2 else key
                                if key.endswith("']") or key.endswith('"]'):
                                    key = key[:-2] if len(key) > 2 else key
                                key = key.strip()
                                cleaned_keys.append(key)
                            else:
                                cleaned_keys.append(key)
                        keys = cleaned_keys

                    keys_str = ", ".join([f"'{key}'" for key in keys])
                    if len(keys) > 1:
                        pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                    else:
                        pyautogui_code.append(f"pyautogui.press({keys_str})")

                elif action == "scroll":
                    pixels = args.get("pixels", 0)
                    pyautogui_code.append(f"pyautogui.scroll({pixels})")

                elif action == "wait":
                    pyautogui_code.append("WAIT")

                elif action == "terminate":
                    # Keep backward-compatible termination token.
                    # (This parser returns a joined string; dict actions would break join.)
                    pyautogui_code.append("DONE")

                elif action == "mouse_move":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(x, y)
                        pyautogui_code.append(
                            f"pyautogui.moveTo({adj_x}, {adj_y})"
                        )
                    else:
                        pyautogui_code.append("pyautogui.moveTo(0, 0)")

                elif action == "left_click_drag":
                    if "coordinate" in args:
                        x, y = args["coordinate"]
                        adj_x, adj_y = adjust_coordinates(x, y)
                        duration = args.get("duration", 0.5)
                        pyautogui_code.append(
                            f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})"
                        )
                    else:
                        pyautogui_code.append("pyautogui.dragTo(0, 0)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Qwen3VL Parser] Failed to parse tool call: {e}")

    # Parse the response
    lines = response.split("\n")
    inside_tool_call = False
    current_tool_call: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract action description
        if line.lower().startswith(("action:")):
            if not low_level_instruction:
                low_level_instruction = line.split("Action:")[-1].strip()
            continue

        # Handle <tool_call> XML tags
        if line.startswith("<tool_call>"):
            inside_tool_call = True
            continue
        elif line.startswith("</tool_call>"):
            if current_tool_call:
                process_tool_call("\n".join(current_tool_call))
                current_tool_call = []
            inside_tool_call = False
            continue

        if inside_tool_call:
            current_tool_call.append(line)
            continue

        # Try to parse standalone JSON (without XML tags)
        if line.startswith("{") and line.endswith("}"):
            try:
                json_obj = json.loads(line)
                if "name" in json_obj and "arguments" in json_obj:
                    process_tool_call(line)
            except json.JSONDecodeError:
                pass

    # Process any remaining tool call
    if current_tool_call:
        process_tool_call("\n".join(current_tool_call))

    # Generate default instruction if not found
    if not low_level_instruction and len(pyautogui_code) > 0:
        if pyautogui_code[0] != "DONE" and pyautogui_code[0] != "WAIT":
            action_type = pyautogui_code[0].split(".", 1)[1].split("(", 1)[0]
            low_level_instruction = f"Performing {action_type} action"
        else:
            low_level_instruction = pyautogui_code[0]

    # Join the list into a single string to match ui_tars_action_parser format
    pyautogui_code_str = "\n".join(pyautogui_code) if pyautogui_code else ""
    return low_level_instruction, pyautogui_code_str