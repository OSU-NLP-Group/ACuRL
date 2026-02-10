import os
import json
import argparse
import glob
import re
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
import datasets
import random

def is_uuid_format(filename):
    """
    Check if filename follows a validation-eligible task-id filename format.
    Historically we used UUID/ow-* only, but some datasets also include valid non-UUID
    task IDs such as "A-01.json" (e.g., KAlgebra/Celestia/ChimeraX/GrassGIS). Some
    ScienceBoard-style names (e.g., "art_slide1.json") are NOT considered
    validation-eligible here.
    Examples: 04578141-1d42-4146-b9cf-6fab4ce5fd74.json -> True
             ow-xlsx-1-11.json -> True
             ow-pptx-2-5.json -> True
             ow-docx-0-8.json -> True
             A-01.json -> True
             art_slide1.json -> False
             category.04578141-1d42-4146-b9cf-6fab4ce5fd74 -> True (extracts filename part)
    """
    # Remove .json extension if present first
    name = filename.replace('.json', '')
    # Handle "category.filename" format: extract filename part (after first dot)
    if '.' in name:
        name = name.split('.', 1)[-1]  # Get part after first dot (category.filename -> filename)
    # UUID pattern: 8-4-4-4-12 hexadecimal characters separated by hyphens
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    # ow-* pattern: ow-xxx-X-Y where xxx is any string and X, Y are numbers
    ow_pattern = r'^ow-\w+-\d+-\d+$'
    # OSWorld "A-01" style pattern (single letter + hyphen + number)
    alpha_dash_num_pattern = r'^[A-Za-z]-\d{1,3}$'
    return bool(
        re.match(uuid_pattern, name, re.IGNORECASE)
        or re.match(ow_pattern, name)
        or re.match(alpha_dash_num_pattern, name)
    )

def load_data(data_dir):
    """
    Load tasks from a directory structure where subdirectories are categories
    and `.json` files are tasks. We only use filenames (no JSON parsing).
    """
    all_data = {}
    categories = []
    
    # Get all category directories
    for category_dir in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category_dir)
        if not os.path.isdir(category_path):
            continue
        
        categories.append(category_dir)
        category_data = []
        
        # Load all json files in this category (recursively search subdirectories)
        json_files = glob.glob(os.path.join(category_path, "**/*.json"), recursive=True)
        for json_file in json_files:
            filename = os.path.splitext(os.path.basename(json_file))[0]  # Remove .json suffix
            task_name = f"{category_dir}.{filename}"  # Add category prefix
            category_data.append(task_name)
        
        all_data[category_dir] = category_data
        print(f"Loaded {len(category_data)} tasks from category: {category_dir}")
    
    return all_data, categories

def load_single_category_from_dir(category_name: str, category_dir: str):
    """
    Load tasks for a single logical category from an explicit directory.
    This is useful when the default data_dir contains multiple duplicated sources
    and you want to constrain
    sampling to a specific subset directory.
    """
    category_data = []
    json_files = glob.glob(os.path.join(category_dir, "**/*.json"), recursive=True)
    for json_file in json_files:
        filename = os.path.splitext(os.path.basename(json_file))[0]  # Remove .json suffix
        task_name = f"{category_name}.{filename}"  # Add category prefix
        category_data.append(task_name)
    print(f"Loaded {len(category_data)} tasks from category override dir: {category_name} -> {category_dir}")
    return category_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument(
        '--data_dir',
        default='./data/tasks/examples',
        help='Path to examples directory',
    )
    parser.add_argument('--train_data_size', default=256, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)
    parser.add_argument('--train_category',nargs='+',default=None,
                        help='Category names to process for training (if not specified, process all categories)')
    parser.add_argument('--val_category', nargs='+', default=None,
                        help='Category names to process for validation (if not specified, use same as training categories)')
    parser.add_argument('--train_category_dir', default=None,
                        help='Optional override directory for loading JSON tasks for the (single) training category. '
                             'When set, the training category tasks are loaded from this directory instead of '
                             '<data_dir>/<train_category>. Validation loading remains unchanged unless --val_category_dir is set.')
    parser.add_argument('--val_category_dir', default=None,
                        help='Optional override directory for loading JSON tasks for the (single) validation category '
                             'when validation has exactly one category. Useful when you want val to come from the same '
                             'subset as train.')
    parser.add_argument(
        '--train_task_pool_file',
        default=None,
        help=(
            'Path to JSON file containing train task IDs for training. If file exists and contains task IDs, '
            'these will be used for train and remaining tasks for val. If not specified or file not found, '
            'train and val will be the same.'
        ),
    )
    parser.add_argument('--task_copy_times', default=8, type=int,
                        help='Number of times to copy each task in the dataset (default: 1, no duplication)')

    args = parser.parse_args()
    print(f"Processing data for mode: {args.mode}")
    print(f"Task copy times: {args.task_copy_times}")

    random.seed(42)

    instruction_following = {
        "visual": "<image>",
        "text": "",
    }

    # Load train task IDs from JSON file
    train_task_ids = None
    if args.train_task_pool_file and os.path.exists(args.train_task_pool_file):
        print(f"Loading train task IDs from: {args.train_task_pool_file}")
        with open(args.train_task_pool_file, "r") as f:
            task_pool_data = json.load(f)
            train_task_ids = task_pool_data["tasks"]
            print(f"Loaded {len(train_task_ids)} train task IDs from JSON file")
    else:
        print(f"Train task pool file not found: {args.train_task_pool_file}")

    print(f"Loading data from: {args.data_dir}")
    all_data, available_categories = load_data(args.data_dir)

    # Optional: override where we load the (single) training category from.
    # IMPORTANT: this should ONLY affect training sampling. Validation should continue to use the
    # canonical examples tree (unless --val_category_dir is explicitly provided).
    train_override_data = None
    train_override_category = None
    if args.train_category_dir:
        if not args.train_category or len(args.train_category) != 1:
            raise ValueError("--train_category_dir requires exactly one training --train_category")
        train_override_category = args.train_category[0]
        if not os.path.isdir(args.train_category_dir):
            raise ValueError(f"--train_category_dir is not a directory: {args.train_category_dir}")
        train_override_data = load_single_category_from_dir(train_override_category, args.train_category_dir)

    # Optional: override where we load the (single) validation category from.
    if args.val_category_dir:
        if not args.val_category or len(args.val_category) != 1:
            raise ValueError("--val_category_dir requires exactly one --val_category")
        val_cat = args.val_category[0]
        if not os.path.isdir(args.val_category_dir):
            raise ValueError(f"--val_category_dir is not a directory: {args.val_category_dir}")
        all_data[val_cat] = load_single_category_from_dir(val_cat, args.val_category_dir)
        if val_cat not in available_categories:
            available_categories.append(val_cat)
    
    # Split into train/val based on train_task_ids from JSON file
    if train_task_ids:
        print(f"Using train task IDs from JSON file: {len(train_task_ids)} tasks")
        train_data_all = {}
        val_data_all = {}
        
        for category, tasks in all_data.items():
            # For training, optionally use an override directory for the (single) training category.
            tasks_for_train = tasks
            if train_override_data is not None and category == train_override_category:
                tasks_for_train = train_override_data

            task_names = list(tasks)
            train_tasks = [t for t in tasks_for_train if t in train_task_ids]
            
            # For validation, keep tasks whose filename matches a validation-eligible ID format.
            val_tasks = [t for t in tasks if is_uuid_format(t)]
            
            train_data_all[category] = train_tasks
            val_data_all[category] = val_tasks
            
            total_tasks = len(task_names)
            uuid_tasks = len([t for t in tasks if is_uuid_format(t)])
            non_uuid_tasks = total_tasks - uuid_tasks
            
            print(f"Category {category}: {len(train_tasks)} tasks for train, {len(val_tasks)} tasks for val")
            print(f"Category {category}: Filtered out {non_uuid_tasks} non-UUID tasks from validation (kept {uuid_tasks} UUID tasks)")
    else:
        print("No train task IDs loaded from JSON file. Using same data for train and val.")
        # Use same data for both train and val, but filter val data for UUID format only
        train_data_all = {}
        val_data_all = {}
        
        for category, tasks in all_data.items():
            # For training, optionally use an override directory for the (single) training category.
            tasks_for_train = tasks
            if train_override_data is not None and category == train_override_category:
                tasks_for_train = train_override_data

            # For train: use all tasks
            train_tasks = list(tasks_for_train)
            # For validation, keep tasks whose filename matches a validation-eligible ID format.
            val_tasks = [t for t in tasks if is_uuid_format(t)]
            
            train_data_all[category] = train_tasks
            val_data_all[category] = val_tasks
            
            total_tasks = len(train_tasks)
            uuid_tasks = len(val_tasks)
            non_uuid_tasks = total_tasks - uuid_tasks
            
            print(f"Category {category}: {len(train_tasks)} tasks for train, {len(val_tasks)} tasks for val")
            print(f"Category {category}: Filtered out {non_uuid_tasks} non-UUID tasks from validation (kept {uuid_tasks} UUID tasks)")
    
    # Process specified categories (or all if not specified)
    train_categories = args.train_category if args.train_category else available_categories
    val_categories = args.val_category if args.val_category else train_categories
    
    print(f"Training categories: {train_categories}")
    print(f"Validation categories: {val_categories}")
    
    train_data = []
    val_data = []
    
    # Process training categories
    for category in train_categories:
        if category not in train_data_all:
            print(f"Warning: Training category '{category}' not found in data. Available categories: {available_categories}")
            continue
            
        category_train_data = train_data_all[category]
        category_train_data = [{"task_name": task} for task in category_train_data]
        
        # Random sample for each category
        sampled_train = random.sample(
            category_train_data,
            k=min(args.train_data_size, len(category_train_data)),
        )
        
        # Copy each task the specified number of times
        if args.task_copy_times > 1:
            sampled_train = sampled_train * args.task_copy_times
            print(f"Category {category}: Each task copied {args.task_copy_times} times")
        
        train_data.extend(sampled_train)
        
        original_train_count = len(sampled_train) // args.task_copy_times if args.task_copy_times > 1 else len(sampled_train)
        
        print(f"Category {category}: Loaded {len(category_train_data)} training examples.")
        print(f"Category {category}: Randomly sampled {original_train_count} unique training examples.")
        print(f"Category {category}: Final dataset contains {len(sampled_train)} training examples (after copying).")
    
    # Process validation categories
    for category in val_categories:
        if category not in val_data_all:
            print(f"Warning: Validation category '{category}' not found in data. Available categories: {available_categories}")
            continue
            
        category_val_data = val_data_all[category]
        category_val_data = [{"task_name": task} for task in category_val_data]
        
        # Random sample for each category
        sampled_val = random.sample(
            category_val_data,
            k=min(args.val_data_size, len(category_val_data)),
        )
        
        # Copy each task the specified number of times
        if args.task_copy_times > 1:
            # sampled_val = sampled_val * args.task_copy_times
            print(f"Category {category}: Each task copied {args.task_copy_times} times")
        
        val_data.extend(sampled_val)
        
        original_val_count = len(sampled_val) // args.task_copy_times if args.task_copy_times > 1 else len(sampled_val)
        
        print(f"Category {category}: Loaded {len(category_val_data)} validation examples.")
        print(f"Category {category}: Randomly sampled {original_val_count} unique validation examples.")
        print(f"Category {category}: Final dataset contains {len(sampled_val)} validation examples (after copying).")
    
    print(f"Total: {len(train_data)} training and {len(val_data)} validation examples across all categories.")
    
    # Print all validation task names for debugging
    print("=== VALIDATION TASK NAMES ===")
    for i, item in enumerate(val_data):
        print(f"Val {i+1}: {item['task_name']}")
    print("=== END VALIDATION TASK NAMES ===")

    # Print all validation task names for debugging
    print("=== TRAIN TASK NAMES ===")
    for i, item in enumerate(train_data):
        print(f"Train {i+1}: {item['task_name']}")
    print("=== END TRAIN TASK NAMES ===")

    # Shuffle only the train data with seed 42 for reproducibility
    print("Shuffling train data with seed 42...")
    random.shuffle(train_data)
    print("Train data shuffling completed.")

    # Load default images
    temp_dataset = datasets.load_dataset('hiyouga/geometry3k')
    default_images = temp_dataset['train'][0]['images']
    
    # Process function (same as original prepare.py)
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = instruction_following[args.mode]
            task_name = example.pop("task_name")

            if args.mode == 'visual':
                data = {
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "task_name": task_name,
                    "images": default_images,
                    "ability": "agent",
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            else:
                data = {
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "task_name": task_name,
                    "ability": "agent",
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            return data
        return process_fn

    # Convert to Dataset and map
    train_dataset = Dataset.from_list(train_data).map(make_map_fn("train"), with_indices=True, num_proc=4)
    val_dataset = Dataset.from_list(val_data).map(make_map_fn("val"), with_indices=True, num_proc=4)

    # Save
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.local_dir, "val.parquet"))

    print(f"Saved datasets to: {args.local_dir}")
    print(f"- train.parquet: {len(train_dataset)} examples")
    print(f"- val.parquet: {len(val_dataset)} examples")

    # Optional HDFS upload
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Uploaded to HDFS: {args.hdfs_dir}")