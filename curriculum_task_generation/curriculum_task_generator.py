import os
import json
import base64
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import openai
from PIL import Image
import io

class SpecificTaskGenerator:
    def __init__(self, openai_api_key: str, model: str = "gpt-4o", enable_diversity: bool = False,
                 software: str = "libreoffice_impress",
                 easy_tasks: int = 1, medium_tasks: int = 1, hard_tasks: int = 1):
        """Initialize specific task generator"""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.enable_diversity = enable_diversity
        self.software = software
        self.easy_tasks = easy_tasks
        self.medium_tasks = medium_tasks
        self.hard_tasks = hard_tasks
        
    def load_trajectory_data(self, trajectory_dir: str) -> Dict[str, Any]:
        """Load trajectory data JSON file from trajectory directory"""
        traj_dir = Path(trajectory_dir)
        
        # Find trajectory JSON file (any JSON file in the directory)
        json_files = list(traj_dir.glob("**/*.json"))
        if not json_files:
            raise FileNotFoundError(f"No trajectory JSON file found in {trajectory_dir}")
        
        trajectory_file = json_files[0]
        print(f"Loading exploration trajectory from: {trajectory_file}")
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        return trajectory_data
    
    def get_screenshots_from_folder(self, screenshots_folder: str, file_pattern: str = "*.png") -> List[Dict[str, Any]]:
        """Get screenshots from folder with specified pattern"""
        import re
        folder = Path(screenshots_folder)
        screenshots = []
        
        # Natural sort function for numeric ordering (e.g., step1.png, step2.png, ..., step10.png)
        def natural_sort_key(path):
            """Natural sort key that handles numbers in filenames"""
            name = path.name
            # Split filename into text and number parts
            parts = re.split(r'(\d+)', name)
            # Convert number parts to int for proper numeric sorting
            return [int(part) if part.isdigit() else part.lower() for part in parts]
        
        # Get PNG files and sort by name (natural sort for numeric ordering)
        png_files = sorted(folder.glob(file_pattern), key=natural_sort_key)
        
        for png_file in png_files:
            # Read image and convert to base64
            with open(png_file, 'rb') as f:
                image_data = f.read()
            
            # Validate image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            screenshots.append({
                'filename': png_file.name,
                'path': str(png_file),
                'base64': base64.b64encode(image_data).decode('utf-8'),
                'size': image.size
            })
                
        
        return screenshots
    
    def get_environment_exploration_screenshots(self, environment_exploration_dir: str) -> List[Dict[str, Any]]:
        """Get environment exploration screenshots (step*.png files)"""
        screenshots = self.get_screenshots_from_folder(environment_exploration_dir, "**/step*.png")
        print(f"Found {len(screenshots)} environment exploration screenshots")
        return screenshots
    
    def get_context_review_screenshots(self, context_review_dir: str) -> List[Dict[str, Any]]:
        """Get context review screenshots from context review directory"""
        screenshots = self.get_screenshots_from_folder(context_review_dir, "*.png")
        print(f"Found {len(screenshots)} context review screenshots")
        return screenshots
    
    def format_environment_exploration_actions(self, trajectory_data: Dict) -> str:
        """Format environment exploration actions for prompt"""
        
        raw_actions = trajectory_data.get('raw_actions', [])
        
        actions_text = "Base Exploration Actions:\n"
        for i, action in enumerate(raw_actions):
            actions_text += f"Step {i+1}: {action}\n"
            
        return actions_text
    
    def _build_content_with_screenshots(self, software_exploration_text: str, environment_exploration_screenshots: List[Dict],
                                       context_review_text: str, context_review_screenshots: List[Dict],
                                       task_requirements_text: str) -> List[Dict]:
        """Build content array with text and screenshots in proper order"""
        content = []
        
        # Part 1: Software Exploration
        content.append({"type": "text", "text": software_exploration_text})
        
        # Add environment exploration screenshots
        for screenshot in environment_exploration_screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot['base64']}",
                    "detail": "high"
                }
            })
        
        # Part 2: Context Review
        content.append({"type": "text", "text": context_review_text})
        
        # Add context review screenshots
        for screenshot in context_review_screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot['base64']}",
                    "detail": "high"
                }
            })
        
        # Part 3: Task Generation Requirements
        content.append({"type": "text", "text": task_requirements_text})
        
        return content
    
    def _call_openai_api(self, content: List[Dict]) -> Optional[str]:
        """Make OpenAI API call with the given content"""
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 16384
        }
        
        # Add temperature for non-reasoning models
        reasoning_models = ["o4-mini", "gpt-5"]
        if self.model not in reasoning_models:
            request_params["temperature"] = 1.0
        
        print(f"Making API call to {self.model} with {len(content)} content items...")
        response = self.client.chat.completions.create(**request_params)
        print(response.choices[0].message.content)
        return response.choices[0].message.content
            

    def load_existing_tasks_for_diversity(self, existing_task_files: List[str], context_filename: str = None, iteration: int = 0) -> List[str]:
        """Load existing tasks for diversity enhancement, optionally filtered by context filename"""
        existing_tasks = []
        
        for task_file in existing_task_files:
            # Filter by context filename if provided and not iteration 0
            if context_filename and iteration > 0 and context_filename not in os.path.basename(task_file):
                continue
                
            if os.path.exists(task_file):
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Read instruction field from OSWorld task config
                if 'instruction' in data:
                    existing_tasks.append(data['instruction'])
        
        if context_filename and iteration > 0:
            print(f"Loaded {len(existing_tasks)} existing tasks from {len(existing_task_files)} files (filtered by '{context_filename}')")
        else:
            print(f"Loaded {len(existing_tasks)} existing tasks from {len(existing_task_files)} files")
        return existing_tasks

    def load_feedback_data(self, feedback_file: str) -> Optional[Dict]:
        """Load feedback data from previous iterations"""
        if not os.path.exists(feedback_file):
            print(f"Warning: Feedback file not found: {feedback_file}")
            return None

        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
        print(f"Loaded feedback data from: {feedback_file}")
        return feedback_data

    def format_feedback_summary(self, feedback_data: Dict, context_filename: str = None) -> str:
        """Format feedback data into a structured summary for the prompt, optionally filtered by context filename"""
        if not feedback_data:
            return ""
        
        task_performances = feedback_data.get('task_performances', [])
        
        if not task_performances:
            return ""
        
        # Build task performance summary with numbered list
        task_performance_lines = []
        for i, task_perf in enumerate(task_performances):
            task_id = task_perf.get('task_id', '')
            avg_sr = task_perf.get('accuracy', 0)
            instruction = task_perf['original_task']["instruction"]
            
            if context_filename in task_id:
                task_performance_lines.append(f"{instruction} (Average SR: {avg_sr:.0%})")
        
        # Format with numbered list like diversity section
        task_performance_text = "\n".join(f"{i+1}. {line}" for i, line in enumerate(task_performance_lines))
        
        feedback_summary = f"""## Feedback on Previous Tasks
Here is a structured summary of learner performance from earlier iterations:

{task_performance_text}"""
        
        return feedback_summary

    def build_iteration_0_prompt(self, environment_exploration_actions: str, environment_exploration_screenshots: List[Dict], 
                                context_review_screenshots: List[Dict], existing_tasks: Optional[List[str]] = None) -> Optional[str]:
        """Build prompt for iteration 0 (no feedback, but with diversity)"""
        
        total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks

        software_exploration_text = f"""You are an expert software task designer. 
Your goal is to generate **exactly {total_tasks} realistic, high-value, and frequently used atomic tasks** for the given software. Each task should represent a **single, minimal yet meaningful operation** within the software — a fundamental action that users frequently perform as part of larger goals.

You will be provided with:
1. A record of random software exploration, including several actions and corresponding screenshots. 
   - This shows the possible functionality of the software.
2. An initial state of the software (with its own screenshots). 
   - This represents the exact starting point from which learners will begin.

## Software Exploration
Actions:
{environment_exploration_actions}

The corresponding {len(environment_exploration_screenshots)} screenshots across the actions: """

        context_review_text = f"""## Initial State
The current initial state is shown in {len(context_review_screenshots)} additional screenshots. These screenshots show the starting point for the software.

The initial state screenshots: """

        # Add diversity section if enabled
        diversity_section = ""
        if self.enable_diversity and existing_tasks:
            diversity_section = f"""## Diversity Requirements
To ensure task diversity, here are some existing tasks to avoid repeating:
{chr(10).join(f"{i+1}. {task}" for i, task in enumerate(existing_tasks))}

- Avoid creating tasks similar to the existing examples above
- Generate creative variations and novel approaches
- Ensure variety in complexity, action types, and user scenarios
- Think of different user personas and use cases"""

        task_requirements_text = f"""## Task Generation Requirements
Based on the random exploration actions and screenshots, combined with the initial state screenshots, generate exactly {total_tasks} **atomic, realistic tasks** that follow these rules:

1. **Atomic and Functional**
   - Each task describes a **single, standalone operation** that can be executed independently.  
   - Do not combine multiple steps or represent a complete workflow.

2. **Anchor in Initial State**  
   Each task must assume the learner starts from the provided initial state screenshots.  
   Do not invent extra elements not visible in the initial state.

3. **Leverage Exploration Evidence**  
   Use the random exploration actions/screenshots to infer what the software can do.  
   Design tasks that are plausible and valuable given those capabilities.  

4. **Clarity and Specificity (Anti-Ambiguity Rule)** 
    All generated tasks must be stated with full clarity and without ambiguity.  
    To ensure this:
    - Do not use vague or illustrative expressions such as “for example”, “such as”, “any”, “multiple”, or other open-ended terms.
    - Do not introduce optionality, alternatives, or flexible ranges (e.g., “A or B”, “choose whichever”, “a few”, “some”).
    - Do not reference hypothetical or representative instances.
    - Every task must define a **single, explicit target** and a **single, unambiguous goal**.
    - The task must be specific enough that a system can determine **exact completion criteria** without guessing.

5. **High-Frequency and High-Value**
   - Focus on **frequently used and pedagogically important operations** — those that represent the core building blocks of real user interactions.

6. **High-Level Expression of Atomicity**
   - Although each task is atomic, it should still reflect a **high-level, meaningful action**, not a low-level procedural step.  
   - Avoid mechanical instructions such as *“click the File menu”* or *“press Ctrl+S.”*  
   - Instead, describe the **intent of the operation** (e.g., “Save the current file,” “Insert an image into the document”).  

7. **Diverse Coverage**
   - The tasks must collectively cover distinct functional categories of the software, not variations of the same operation.  
   - Ensure diversity across:
     - **Functional intents** (e.g., creation, editing, formatting, navigation, organization, configuration, importing, exporting).  
     - **UI regions** visible in the screenshots (e.g., toolbar, sidebar, menus, panels, workspace, dialogs).  
     - **Object types** present in the software (e.g., text elements, files, sheets, slides, images, fields, widgets).  
     - **User personas** (e.g., content creation, review, organization, adjustment, inspection).  
   - Avoid producing tasks that differ only superficially or that belong to the same narrow action category.  
   - Each task should introduce a unique, non-overlapping operation that explores a different aspect of the software’s capabilities.

{diversity_section}

Return the tasks in the following JSON structure:

```json
{{
    "tasks": [
        {{
            "task_id": "task_1",
            "task_description": "An atomic software operation achievable from the given initial state"
        }},
        {{
            "task_id": "task_2",
            "task_description": "An atomic software operation achievable from the given initial state"
        }},
        {{
            "task_id": "task_3",
            "task_description": "An atomic software operation achievable from the given initial state"
        }},
        {{
            "task_id": "task_4", 
            "task_description": "An atomic software operation achievable from the given initial state"
        }},
        ......
    ]
}}
```"""
        print(task_requirements_text)
        content = self._build_content_with_screenshots(
            software_exploration_text=software_exploration_text,
            environment_exploration_screenshots=environment_exploration_screenshots,
            context_review_text=context_review_text,
            context_review_screenshots=context_review_screenshots,
            task_requirements_text=task_requirements_text,
        )
        return self._call_openai_api(content)

    def build_iteration_1plus_prompt(self, environment_exploration_actions: str, environment_exploration_screenshots: List[Dict], 
                                    context_review_screenshots: List[Dict], existing_tasks: Optional[List[str]] = None, 
                                    feedback_data: Optional[Dict] = None, context_filename: str = None) -> Optional[str]:
        """Build prompt for iteration 1+ (with feedback and diversity)"""
        
        total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks

        software_exploration_text = f"""You are a professional instructional designer specializing in software learning task design. 
Your objective is to generate exactly new {total_tasks} high-quality, realistic learning tasks based on the previous feedback.

You will be provided with:
1. A record of random software exploration, including several actions and corresponding screenshots. 
    - This shows the possible functionality of the software.
2. An initial state of the software (with its own screenshots). 
    - This represents the exact starting point from which learners will begin.
3. A summary of learner performance on earlier tasks.  
    - This indicates which tasks were difficult, what the learner can already solve, and what gaps or weaknesses remain.

## Software Exploration
Actions:
{environment_exploration_actions}

The corresponding {len(environment_exploration_screenshots)} screenshots across the actions: """

        context_review_text = f"""## Initial State
The current initial state is shown in {len(context_review_screenshots)} additional screenshots. These screenshots show the starting point for the software.

The initial state screenshots: """

        # Add feedback section (always present for iteration 1+)
        feedback_section = self.format_feedback_summary(feedback_data, context_filename)

        # Add diversity section if enabled
        diversity_section = ""
        if self.enable_diversity and existing_tasks:
            diversity_section = f"""## Diversity Requirements
To ensure task diversity, here are some existing tasks to avoid repeating:
{chr(10).join(f"{i+1}. {task}" for i, task in enumerate(existing_tasks))}

- Avoid creating tasks similar to the existing examples above
- Generate creative variations and novel approaches
- Ensure variety in complexity, action types, and user scenarios
- Think of different user personas and use cases"""

        task_requirements_text = f"""## Task Generation Requirements
Based on the random exploration actions and screenshots, combined with the initial state screenshots, generate exactly {total_tasks} specific tasks that:

1. **High-Level User Goals (NO Step-by-Step Instructions Allowed)**  
    - Write tasks as **natural user goals**, NOT tutorials or procedures.  
    - Express tasks the way a real user would describe what they want to achieve.  
    - Absolutely **do NOT describe UI navigation paths** (e.g., “click X”, “open Y”, “go to View > Animation”).  
    - No operational verbs tied to UI mechanics (e.g., “open”, “select”, “navigate to”, “choose”).  
    - Use **intent-oriented phrasing**, not action-oriented phrasing.  
    - Each task must be **one or two concise sentences** describing the desired outcome or goal.

2. **Anchor in Initial State**  
    - Each task must assume the learner starts from the provided initial state screenshots.  
    - Do not invent extra elements not visible in the initial state.  

3. **Leverage Exploration Evidence**  
    - Use the random exploration actions/screenshots to infer what the software can do.  
    - Design tasks that are plausible and valuable given those capabilities.  

4. **Realistic User Value**  
    - Tasks should reflect meaningful activities a real user of this software might want to accomplish, not artificial button-clicking exercises.    

5. **Distinct Coverage**  
    - Each of the task should focus on a different aspect, function, or workflow of the software.  
    - Avoid overlap. Aim for variety (e.g., editing, organizing, exporting, formatting).   

6. **Diversity**
    - Avoid creating tasks similar to the existing ones from previous feedback.
    - Generate creative variations and novel approaches.
    - Ensure variety in complexity, action types, and user scenarios.
    - Think of different user personas and use cases.

7. **Clarity and Specificity (Anti-Ambiguity Rule)**  
    All regenerated tasks must be stated with full clarity and without ambiguity.  
    To ensure this:
    - Do not use vague or illustrative expressions such as “for example”, “such as”, “any”, “multiple”, or other open-ended terms.
    - Do not introduce optionality, alternatives, or flexible ranges (e.g., “A or B”, “choose whichever”, “a few”, “some”).
    - Do not reference hypothetical or representative instances.
    - Every task must define a **single, explicit target** and a **single, unambiguous goal**.
    - The task must be specific enough that a system can determine **exact completion criteria** without guessing.

8. **Feedback Integration**  
    You must generate a substantively different new task based on the agent’s SR.
    The new task must not be a minor revision, paraphrase, or parameter change of the original task.

    #### **If SR > 70% (High performance)**
    The agent performs the original task reliably. You must therefore generate a **fully new** task that is **significantly more challenging**, with **no structural, semantic, or phrasing resemblance** to the original.

    The new task must satisfy all of the following:

    **1. Fully New Scenario**
    - Use a **completely different real user motivation**, context, and type of content.
    - Do NOT share the original task's goal type, UI element category, action verbs, workflow pattern, or problem structure.

    **2. Higher Complexity**
    The new task must be **meaningfully more complex**, for example by:
    - involving **multiple related sub-goals**, 
    - introducing **conditional or constraint-based goals**, 
    - requiring **cross-region or cross-document coordination**, 
    - adding **explicit constraints, requirements, or trade-offs**, 
    - or requiring **richer interpretation or reasoning** about the interface/content.

    **3. Strict Non-Similarity**
    The new task must NOT:
    - resemble the original in structure or step pattern,
    - reuse similar keywords, verbs, UI components, or operations,
    - operate in the same functional space as the original (e.g., if original used formatting, avoid formatting),
    - pursue any version of the same underlying objective.

    The final task must feel like an **entirely new, high-value, realistic scenario** that a different user might genuinely perform.

    #### **If 30% ≤ SR ≤ 70% (Medium performance)**
    The agent shows partial competence. You must generate a **different task of similar overall difficulty**, but with a **distinct scenario and distinct user intent**.

    You may:
    - explore a **new real user scenario**, or
    - create a **lateral variation** that practices similar reasoning skills without repeating the original pattern.

    Ensure meaningful variation by changing at least one of:
    - the **user persona**,
    - the **purpose or motivation**,
    - the **content being worked on**,
    - or the **overall context** (document type, audience, situation).

    The new task must **not** be a softened or strengthened rewrite of the original.

    #### **If SR < 30% (Low performance)**
    When the agent struggles significantly, generate a **scaffolded prerequisite task** that focuses on the fundamental skill(s) required to perform the original task. 

    This scaffolded task must:
    - isolate **one essential underlying capability** needed to accomplish the original task,
    - be formulated as an **independent, authentic user goal**, not a simplified or partially completed version of the original task,
    - introduce a **distinct scenario, purpose, or context**, while still remaining within the same broad capability domain,
    - and explicitly target the foundational competencies the agent lacks (e.g., identifying relevant elements, applying targeted adjustments, interpreting context, selecting appropriate options, or manipulating core components).

    The new task should reflect a **core sub-skill** that the user must master before attempting any higher-level or more complex versions of the original task.

{feedback_section}

Return the tasks in the following JSON structure:

```json
{{
    "tasks": [
        {{
            "task_id": "task_1",
            "original_task_description": "The original task description from the previous feedback",
            "reasoning": "Explain why and how this task was modified given the agent's SR and feedback context.",
            "new_task_description": "High-level yet specific user goal derived from the initial state and refined through feedback."
        }},
        {{
            "task_id": "task_2",
            "original_task_description": "The original task description from the previous feedback",
            "reasoning": "Explain why and how this task was modified given the agent's SR and feedback context.",
            "new_task_description": "High-level yet specific user goal derived from the initial state and refined through feedback."
        }},
        {{
            "task_id": "task_3",
            "original_task_description": "The original task description from the previous feedback",
            "reasoning": "Explain why and how this task was modified given the agent's SR and feedback context.",
            "new_task_description": "High-level yet specific user goal derived from the initial state and refined through feedback."
        }},
        {{
            "task_id": "task_4",    
            "original_task_description": "The original task description from the previous feedback",
            "reasoning": "Explain why and how this task was modified given the agent's SR and feedback context.",
            "new_task_description": "High-level yet specific user goal derived from the initial state and refined through feedback."
        }},
        ......
    ]
}}
```"""
        
        print(task_requirements_text)
        content = self._build_content_with_screenshots(
            software_exploration_text, environment_exploration_screenshots,
            context_review_text, context_review_screenshots,
            task_requirements_text
        )
        return self._call_openai_api(content)


    def generate_specific_tasks(self, environment_exploration_actions: str, environment_exploration_screenshots: List[Dict],
                              context_review_screenshots: List[Dict], existing_tasks: Optional[List[str]] = None, 
                              feedback_data: Optional[Dict] = None, iteration: int = 0, context_filename: str = None) -> Optional[str]:
        """Generate specific tasks using GPT models - dispatches to appropriate prompt builder"""
        
        # If no feedback data provided, use iteration 0 prompt
        if not feedback_data or iteration == 0:
            return self.build_iteration_0_prompt(
                environment_exploration_actions=environment_exploration_actions,
                environment_exploration_screenshots=environment_exploration_screenshots,
                context_review_screenshots=context_review_screenshots,
                existing_tasks=existing_tasks,
            )
        
        # Otherwise use iteration 1+ prompt with feedback
        return self.build_iteration_1plus_prompt(
            environment_exploration_actions=environment_exploration_actions,
            environment_exploration_screenshots=environment_exploration_screenshots,
            context_review_screenshots=context_review_screenshots,
            existing_tasks=existing_tasks,
            feedback_data=feedback_data,
            context_filename=context_filename,
        )
    
    def create_task_config_file(self, task_data: Dict,
                               context_filename: str, output_dir: str,
                               context_template: Optional[Dict[str, Any]] = None) -> str:
        """Create OSWorld-compatible task configuration file"""
        
        task_id = task_data['task_id']
        task_description = task_data.get('task_description', task_data.get('new_task_description', ''))
        original_task_description_raw = task_data.get('original_task_description')
        if original_task_description_raw:
            original_task_description = original_task_description_raw.split(" (Average SR")[0]
        else:
            original_task_description = None
        software_name = context_template.get('snapshot', ['unknown']) if isinstance(context_template, dict) else ['unknown']
        reasoning = task_data.get('reasoning', '')
        
        # Generate meaningful filename based on context and task.
        filename = f"{context_filename}_{task_id}.json"
        
        # Create meaningful task ID combining context filename and task ID
        full_task_id = f"{context_filename}_{task_id}"

        config = json.loads(json.dumps(context_template))

        config["id"] = full_task_id
        config["snapshot"] = config.get("snapshot", software_name)
        config["instruction"] = task_description
        config["source"] = config.get("source", "")

        # Optional metadata for iterative refinement (kept if provided)
        if original_task_description:
            config["original_task"] = original_task_description
        if reasoning:
            config["reasoning"] = reasoning
        
        # Save config file
        output_path = Path(output_dir) / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"Created task config: {output_path}")
        return str(output_path)
    
    def update_task_index_json(self, generated_task_files: List[str], context_filename: str, output_dir: str, model: str = None, index_dir: str = None) -> str:
        """Create or update a task-index JSON file that lists all generated task IDs"""
        
        # Extract task IDs from generated files
        new_task_ids = []
        for task_file in generated_task_files:
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                    task_id = task_data.get('id')
                    if task_id:
                        new_task_ids.append(f"{self.software}.{task_id}")
            except Exception as e:
                print(f"Warning: Could not read task file {task_file}: {e}")
        
        # Default naming
        task_index_filename = f"{self.software}.generated_task.json"
        if model:
            task_index_filename = f"{self.software}.generated_task_{model}.json"
        
        # Create target directory path - use index_dir if provided
        if not index_dir:
            raise ValueError("index_dir must be provided")
        target_dir = index_dir
        
        os.makedirs(target_dir, exist_ok=True)
        task_index_path = Path(target_dir) / task_index_filename
        
        # Load existing task index if file exists
        existing_task_ids = []
        if task_index_path.exists():
            try:
                with open(task_index_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_task_ids = existing_data.get('tasks', [])
                print(f"Loaded {len(existing_task_ids)} existing task IDs from task index file")
            except Exception as e:
                print(f"Warning: Could not read existing task index file: {e}")
        
        # Merge new task IDs with existing ones (avoid duplicates)
        all_task_ids = existing_task_ids.copy()
        for task_id in new_task_ids:
            if task_id not in all_task_ids:
                all_task_ids.append(task_id)
        
        # Create updated task index data structure
        task_index_data = {
            "tasks": sorted(all_task_ids)  # Sort for consistent ordering
        }
        
        # Save updated task index file
        with open(task_index_path, 'w', encoding='utf-8') as f:
            json.dump(task_index_data, f, ensure_ascii=False, indent=2)
        
        print(f"Updated task index JSON: {task_index_path}")
        print(f"Added {len(new_task_ids)} new task IDs, total: {len(all_task_ids)} task IDs")
        return str(task_index_path)
    
    def generate_tasks_from_context(self, 
                                         environment_exploration_dir: str,
                                         context_review_dir: str,
                                         context_filename: str,
                                         output_dir: str,
                                         index_dir: str,
                                         existing_task_files: List[str] = None,
                                         feedback_file: str = None,
                                         iteration: int = 0,
                                         context_template: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate tasks from context review
        
        Args:
            environment_exploration_dir: Directory containing trajectory data (trajectory directory)
            context_review_dir: Directory containing initial state screenshots (Context Review)
            context_filename: Name of the context file (e.g., technology_slide1)
            output_dir: Output directory for generated task files
            index_dir: Directory for index file (task pool file)
            existing_task_files: List of existing task files for diversity enhancement
            feedback_file: Path to feedback file from previous iterations
            iteration: Current iteration number
            context_template: Template for context task configuration
        """
        
        print(f"Generating tasks from context: {context_filename}")
        
        # Load trajectory data from trajectory directory
        trajectory_data = self.load_trajectory_data(environment_exploration_dir)
        
        # Get environment exploration screenshots from trajectory directory
        environment_exploration_screenshots = self.get_environment_exploration_screenshots(environment_exploration_dir)
        
        # Get context review screenshots
        context_review_screenshots = self.get_context_review_screenshots(context_review_dir)
        
        if not context_review_screenshots:
            print("Warning: No context review screenshots found")
            return []
        
        # Format environment exploration actions
        environment_exploration_actions = self.format_environment_exploration_actions(trajectory_data)
        
        # Use context_template as base_config if available, otherwise create a minimal one
        base_config = context_template
        
        # Load existing tasks for diversity if enabled
        existing_tasks = None
        if self.enable_diversity and existing_task_files:
            existing_tasks = self.load_existing_tasks_for_diversity(existing_task_files, context_filename, iteration)
            print(f"Loaded {len(existing_tasks)} existing tasks for diversity enhancement")
        
        # Load feedback data if provided, otherwise use iteration 0
        feedback_data = None
        if feedback_file:
            feedback_data = self.load_feedback_data(feedback_file)
            if feedback_data:
                print(f"Loaded feedback data for adaptive task generation")
            else:
                # If feedback file doesn't exist or is invalid, use iteration 0
                iteration = 0
        else:
            # No feedback file provided, use iteration 0
            iteration = 0
        
        # Generate tasks with minimal retry if count mismatches desired total
        desired_total_tasks = self.easy_tasks + self.medium_tasks + self.hard_tasks
        max_retries = 3
        attempt_index = 0
        tasks = []
        while attempt_index < max_retries:
            print("Generating specific tasks based on initial state...")
            tasks_response = self.generate_specific_tasks(
                environment_exploration_actions=environment_exploration_actions,
                environment_exploration_screenshots=environment_exploration_screenshots,
                context_review_screenshots=context_review_screenshots,
                existing_tasks=existing_tasks,
                feedback_data=feedback_data,
                iteration=iteration,
                context_filename=context_filename,
            )
            if not tasks_response:
                print("Failed to generate tasks response, retrying..." if attempt_index < max_retries - 1 else "Failed to generate tasks")
                attempt_index += 1
                continue
            # Parse generated tasks
            # Clean the response - remove markdown code blocks if present
            cleaned_response = tasks_response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            try:
                tasks_data = json.loads(cleaned_response)
                tasks = tasks_data.get('tasks', [])
                if len(tasks) != desired_total_tasks:
                    print(f"Warning: Expected {desired_total_tasks} tasks, got {len(tasks)}. Retrying..." if attempt_index < max_retries - 1 else f"Warning: Expected {desired_total_tasks} tasks, got {len(tasks)}.")
                    attempt_index += 1
                    if attempt_index < max_retries:
                        continue
                # Replace placeholder task_ids with actual UUIDs
                for i, task in enumerate(tasks):
                    if 'task_id' in task and isinstance(task['task_id'], str) and task['task_id'].startswith('task_'):
                        task['task_id'] = f"task_{str(uuid.uuid4())[:16]}"
                break
            except json.JSONDecodeError as e:
                print(f"Error parsing tasks JSON: {e}")
                print(f"Cleaned response: {cleaned_response}")
                attempt_index += 1
                if attempt_index < max_retries:
                    print("Retrying...")
                    continue
                return []
        if not tasks:
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create task configuration files
        generated_files = []
        for task in tasks:
            config_file = self.create_task_config_file(
                task_data=task,
                context_filename=context_filename,
                output_dir=output_dir,
                context_template=context_template,
            )
            generated_files.append(config_file)
        
        print(f"Generated {len(generated_files)} specific task configuration files")
        
        # Generate task index JSON file
        task_index_file = self.update_task_index_json(
            generated_task_files=generated_files,
            context_filename=context_filename,
            output_dir=output_dir,
            model=self.model,
            index_dir=index_dir,
        )
        if task_index_file:
            print(f"Generated task index JSON file: {task_index_file}")
        
        return generated_files

def process_all_tasks(generator, args, software):
    """Process all tasks in the context review folder"""
    context_review_dir = Path(args.context_review_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all context review directories matching pattern
    context_pattern = f"{software}.*"
    context_review_dirs = list(context_review_dir.glob(context_pattern))
    
    if not context_review_dirs:
        print(f"No context review directories found matching pattern: {context_pattern}")
        return
    
    print(f"🔍 Found {len(context_review_dirs)} context review directories")
    print(f"📁 Context review directory: {context_review_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"🎯 Diversity enabled: {args.enable_diversity}")
    print(f"📝 Software: {software}")
    # exploration file naming is derived from software (+ optional model); no custom override.
    
    # Statistics
    context_count = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process each context review directory
    for context_review_item_dir in sorted(context_review_dirs):
        if not context_review_item_dir.is_dir():
            continue
            
        context_count += 1
        context_review_name = context_review_item_dir.name
        print(f"📋 Processing context review {context_count}: {context_review_name}")
        
        # Find the training_step_0 directory
        training_dir = context_review_item_dir / "training_step_0"
        if not training_dir.exists():
            print(f"  ⚠️  Warning: No training_step_0 directory found in {context_review_name}")
            failed_count += 1
            continue
        
        # Find the UID directory (should be the only subdirectory)
        uid_dirs = [d for d in training_dir.iterdir() if d.is_dir() and len(d.name.split('-')) == 5]
        if not uid_dirs:
            print(f"  ⚠️  Warning: No UID directory found in {context_review_name}/training_step_0")
            failed_count += 1
            continue
        
        context_review_path = str(uid_dirs[0])
        
        # Extract the base filename from context review name
        context_filename = context_review_name.replace(f"{software}.", "")
        
        # Get the real filename from the corresponding JSON file in examples directory
        tasks_root = Path(f'./data/tasks/examples/{software}')
        context_task_json = tasks_root / "context_review" / f"{context_filename}.json"
        context_template = None
        if context_task_json.exists():
            with open(context_task_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                context_template = data
        
        print(f"  📸 Context review path: {context_review_path}")
        print(f"  📄 Initial state file: {context_filename}")
        
        # Check if this task has already been processed.
        # skipping different ids that share the same context_filename.
        existing_task_files = list(output_dir.glob(f"{context_filename}_task_*.json"))
        
        if existing_task_files:
            print(f"  ⏭️  Skipping {context_review_name} - already processed ({len(existing_task_files)} task files found)")
            skipped_count += 1
            continue
        
        # Get existing tasks for diversity if enabled
        existing_task_files = []
        if args.enable_diversity:
            # Get existing tasks from both existing_tasks_dir and current output_dir
            all_existing_dirs = []
            
            # Add existing_tasks_dir if provided
            if args.existing_tasks_dir:
                existing_tasks_dir = Path(args.existing_tasks_dir)
                if existing_tasks_dir.exists():
                    all_existing_dirs.append(existing_tasks_dir)
            
            # Collect task files from all directories (recursively search subdirectories)
            for existing_dir in all_existing_dirs:
                # Find all existing task files recursively
                task_files = list(existing_dir.glob("**/*_task_*.json"))
                existing_task_files.extend([str(f) for f in task_files])
            
            if existing_task_files:
                print(f"  🎯 Using diversity enhancement with {len(existing_task_files)} existing task files")
                print(f"  📁 Scanned directories: {[str(d) for d in all_existing_dirs]}")
        
        # Run the specific task generator for this context review
        print(f"  🚀 Running generator...")
        
        try:
            generated_files = generator.generate_tasks_from_context(
                environment_exploration_dir=str(args.environment_exploration_dir),
                context_review_dir=context_review_path,
                context_filename=context_filename,
                output_dir=str(output_dir),
                index_dir=args.index_dir,
                existing_task_files=existing_task_files,
                feedback_file=getattr(args, 'feedback_file', None),
                iteration=getattr(args, 'iteration', 0),
                context_template=context_template,
            )
            
            if generated_files:
                print(f"  ✅ Context review {context_review_name} completed successfully!")
                success_count += 1
            else:
                print(f"  ❌ Context review {context_review_name} failed - no files generated!")
                failed_count += 1
                
        except Exception as e:
            print(f"  ❌ Context review {context_review_name} failed with error: {str(e)}")
            failed_count += 1
    
    # Summary
    print("📊 Processing Summary:")
    print(f"  Total context reviews found: {context_count}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (already processed): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  🎯 Diversity enabled: {args.enable_diversity}")
    print(f"  📝 Software: {software}")

def main():
    parser = argparse.ArgumentParser(description='Generate specific learning tasks from initial state')
    parser.add_argument('--environment-exploration-dir', required=True, help='Directory containing trajectory data (trajectory directory)')
    parser.add_argument('--context-review-dir', required=True, help='Directory containing task directories with initial state screenshots (Context Review)')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for generated task files')
    parser.add_argument('--index-dir', required=True, help='Directory for index file (task pool file)')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--model', required=True, help='GPT model to use (e.g., gpt-4o, gpt-3.5-turbo)')
    parser.add_argument('--enable-diversity', required=True, help='Enable diversity enhancement using existing tasks')
    parser.add_argument('--existing-tasks-dir', help='Directory containing existing task files for diversity enhancement')
    parser.add_argument('--software', required=True, help='Software name (e.g., libreoffice_impress)')
    parser.add_argument('--easy-tasks', type=int, default=16, help='Number of easy tasks to generate (default: 1)')
    parser.add_argument('--medium-tasks', type=int, default=10, help='Number of medium tasks to generate (default: 1)')
    parser.add_argument('--hard-tasks', type=int, default=6, help='Number of hard tasks to generate (default: 1)')
    parser.add_argument('--feedback-file', help='Path to feedback file from previous iterations for adaptive task generation')
    parser.add_argument('--iteration', type=int, default=0, help='Current iteration number (0 for first iteration, 1+ for subsequent iterations). If --feedback-file is not provided, iteration will be set to 0.')
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.environment_exploration_dir):
        print(f"Error: Environment exploration directory {args.environment_exploration_dir} does not exist")
        return
        
    if not os.path.exists(args.context_review_dir):
        print(f"Error: Context review directory {args.context_review_dir} does not exist")
        return
    
    # Create specific task generator
    generator = SpecificTaskGenerator(args.api_key, args.model, args.enable_diversity, 
                                    args.software,
                                    args.easy_tasks, args.medium_tasks, args.hard_tasks)
    
    # Process all tasks in the task folder
    process_all_tasks(generator, args, args.software)

if __name__ == "__main__":
    main()
