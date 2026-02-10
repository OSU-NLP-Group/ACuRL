import asyncio
import base64
import re
import io
from PIL import Image
from .utils import OpenaiEngine, vllm_OpenaiEngine

identify_key_points_system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
- A **key point** is a critical element, condition, requirement, constraint, or step explicitly mentioned in the task description.
- Do not infer or add any unstated elements.
- Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""

identify_key_screenshot_system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
### Reasoning: [Your explanation]  
### Score: [1-5]"""

outcome_system_msg = """You are an expert evaluator of computer use agents. Your task is to rigorously assess whether an agent has successfully completed a given task by examining the initial state, final state, action history, and intermediate screenshots.

## Evaluation Process

Follow this systematic evaluation process:

### Step 1: Understand the Task Requirements
- Read the task description carefully and identify what needs to be accomplished
- Review the key points that define successful completion
- Examine the initial screenshot(s) to understand the starting state

### Step 2: Compare Initial vs Final State
- **Critical**: Compare the initial screenshot(s) with the final screenshot side-by-side
- Identify what specific changes occurred in the interface
- Determine if these changes align with the task requirements
- Check if the final state demonstrates completion of all required modifications

### Step 3: Verify Each Key Point
For EACH key point listed, systematically check:
- Is this requirement explicitly addressed in the action history?
- Is the result visible in the final screenshot or intermediate screenshots?
- Is the implementation correct (not just superficially similar)?
- Are the exact specifications met (correct values, correct targets, correct methods)?

### Step 4: Apply Strict Evaluation Criteria
1: The agent must only make changes that are directly necessary to complete the task. Any unrelated edits or system/software modifications that are not explicitly required for task completion will result in failure.
2: Interpret the user's instructions in the context of the current software. Professional or common software terms have precise meanings in their respective applications. An action that superficially matches the wording but targets the wrong function or interface region of the software should be considered a failure.
3: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
4: If the agent loops through a sequence of actions that do not make progress toward the goal (including failing to click "Save" or "Submit," etc.), it should be considered a failure.


### Step 5: Make Final Judgment

**SUCCESS** requires:
- ALL key points are satisfied
- ALL evaluation criteria are met
- Final screenshot shows correct end state
- No extraneous or incorrect modifications

**FAILURE** if ANY of these are true:
- Even one key point is not satisfied
- Any evaluation criterion is violated
- Final state does not match requirements
- Evidence of incorrect or incomplete execution

## Response Format

Provide your evaluation in exactly two lines:

Thoughts: [Systematically walk through Steps 1-5 above. For each key point, explicitly state whether it's satisfied and cite evidence from screenshots/actions. Compare initial and final states. Note any violations of criteria.]
Status: "success" or "failure"

**Important**: Be thorough but concise. Your thoughts should demonstrate you checked each key point and criterion."""


class CUAJudgeEvaluator:
    def __init__(self, key_identification_screenshot_model = "o4-mini", key_points_outcome_model = "o4-mini", max_image=50, score_threshold=3, use_vllm_for_key_screenshot=False, vllm_base_url=None):
        self.max_image = max_image
        self.score_threshold = score_threshold
        
        
        if use_vllm_for_key_screenshot:
            assert vllm_base_url is not None, "vllm_base_url must be provided if use_vllm_for_key_screenshot is True"

        if not use_vllm_for_key_screenshot:
            self.key_identification_screenshot_model = OpenaiEngine(model = key_identification_screenshot_model)
        else:
            self.key_identification_screenshot_model = vllm_OpenaiEngine(model = key_identification_screenshot_model, base_url = vllm_base_url)
            
        self.key_points_outcome_model = OpenaiEngine(model = key_points_outcome_model)

    def encode_image(self, image):
        """Convert a PIL image to base64 string."""
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    async def get_score(self, response):
            reward = None
            try:
                if "success" in response.lower().split('status:')[1]:
                    reward = 1.0
                else:
                    reward = 0.0
            except:
                reward = 0.0

            return reward

    async def identify_key_points(self, task, input_image_paths):
        prompt = """Task: {task}"""
        text = prompt.format(task=task)

        input_images_msg = []

        if input_image_paths != None:
            for input_image_path in input_image_paths:
                input_images_jpg_base64_str = self.encode_image(Image.open(input_image_path))
                input_images_msg.append(
                                            {
                                                'type': 'image_url',
                                                'image_url': {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                                            }
                                        )
        messages = [
                        {
                            "role": "system", 
                            "content": identify_key_points_system_msg
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": text}] + input_images_msg,
                        }
                    ]
        responses = await asyncio.to_thread(self.key_points_outcome_model.generate, messages)
        return responses[0]

    async def judge_image(self, task, input_image_paths, image_path, key_points):
        prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
        text = prompt.format(task=task,key_points=key_points)

        input_images_msg = []
        if input_image_paths != None:
            for input_image_path in input_image_paths:
                input_images_jpg_base64_str = self.encode_image(Image.open(input_image_path))
                input_images_msg.append(
                                            {
                                                'type': 'image_url',
                                                'image_url': {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                                            }
                                        )
        messages = [{"role": "system", "content": identify_key_screenshot_system_msg}]

        if input_images_msg:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "The input images are:"}] + input_images_msg
            })
        
        jpg_base64_str = self.encode_image(Image.open(image_path))
        messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}", "detail": "high"},
                        },
                    ]
                }
            )

        responses = await asyncio.to_thread(self.key_identification_screenshot_model.generate, messages)
        return responses[0]


    async def cuajudge_general_eval(self, task, input_image_paths, action_thoughts, last_actions, images_path):
        prompt = """## Task to Evaluate
{task}

## Key Points for Successful Completion
{key_points}

## Agent's Action History
{last_actions}

## Important Intermediate Screenshots and Analysis
The following screenshots were identified as potentially important in the agent's trajectory:
{thoughts}

---
Now, evaluate whether the agent successfully completed the task following the systematic evaluation process outlined in your instructions."""


        key_points = await self.identify_key_points(task, input_image_paths)
        key_points = key_points.replace("\n\n", "\n")

        try:
            key_points = key_points.split("**Key Points**:")[1]
            key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
        except:
            key_points = key_points.split("Key Points:")[-1]
            key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

        tasks = [self.judge_image(task, input_image_paths, image_path, key_points) for image_path in images_path]
        image_responses = await asyncio.gather(*tasks)

        input_images_msg = []
        whole_content_img = []
        whole_thoughts = []
        record = []
        pattern = r"[1-5]"
        for response, image_path in zip(image_responses, images_path):
            try:
                score_text = response.split("### Score")[1]
                thought = response.split("### Reasoning:")[-1].strip().lstrip("\n").split("### Score")[0].replace('\n',' ')
                score = re.findall(pattern, score_text)[0]
                record.append({"Response": response, "Score": int(score)})
            except Exception as e:
                print(f"Error processing response: {e}")
                score = 0
                record.append({"Response": response, "Score": 0})

            if int(score) >= self.score_threshold:
                jpg_base64_str = self.encode_image(Image.open(image_path))
                whole_content_img.append(
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"}
                    }
                )
                if thought != "":
                    whole_thoughts.append(thought)

        whole_content_img = whole_content_img[:self.max_image]
        whole_thoughts = whole_thoughts[:self.max_image]
        if len(whole_content_img) == 0:
            prompt = """## Task to Evaluate
{task}

## Key Points for Successful Completion
{key_points}

## Agent's Action History
{last_actions}

## Important Intermediate Screenshots and Analysis
No intermediate screenshots were flagged as particularly important. Please rely on the initial and final screenshots, along with the action history, to evaluate task completion.

---
Now, evaluate whether the agent successfully completed the task following the systematic evaluation process outlined in your instructions."""

        if action_thoughts != None:
            text = prompt.format(task=task, last_actions="\n".join(f"{i+1}. {action}. Reasoning: {action_thought}" for i, (action, action_thought) in enumerate(zip(last_actions,action_thoughts))), key_points=key_points, thoughts = "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)))

        else:
            text = prompt.format(task=task, last_actions="\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions)), key_points=key_points, thoughts = "\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)))

        input_images_msg = []
        if input_image_paths is not None:
            for path in input_image_paths:
                input_images_jpg_base64_str = self.encode_image(Image.open(path))
                input_images_msg.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{input_images_jpg_base64_str}", "detail": "high"}
                })

        # Get final screenshot
        final_screenshot_msg = []
        if images_path and len(images_path) > 0:
            final_image_path = images_path[-1]
            final_jpg_base64_str = self.encode_image(Image.open(final_image_path))
            final_screenshot_msg.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{final_jpg_base64_str}", "detail": "high"}
            })

        messages = [{"role": "system", "content": outcome_system_msg}]

        if input_images_msg:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "The initial screenshot(s) are:"}] + input_images_msg
            })

        if final_screenshot_msg:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": "The final screenshot is:"}] + final_screenshot_msg
            })

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}] + whole_content_img
        })
        response = self.key_points_outcome_model.generate(messages)
        reward = await self.get_score(response[0])

        # Return additional information for detailed logging
        return response[0], reward, {
            'key_points': key_points,
            'thoughts': whole_thoughts,
            'image_judge_record': record
        }
