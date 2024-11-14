####################################################################
# Evaluation File for Vectorisation vs Upscaling
####################################################################


# Importing Libraries
import os
import base64
import json
import requests
from datetime import datetime
from typing import Any, Optional
from braintrust import Eval, Score, load_prompt
from dotenv import load_dotenv
import braintrust
from braintrust import wrap_openai
from openai import AsyncOpenAI
import imghdr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv('.env')

dataset = braintrust.init_dataset(
    project="Vectorisation vs Upscaling",
    name="Vectorisation vs Upscaling"
)

experiment = braintrust.init(
    project="Vectorisation vs Upscaling",
    dataset=dataset
)

prompt = load_prompt(
    project="Vectorisation vs Upscaling",
    slug="vectorisation-vs-upscaling-4222",
    version="35002287caf1767e"
)

temperatures = [0]
aggregated_results = {
    "total_samples": 0,
    "correct_predictions": 0,
    "class_metrics": {
        "vectorizing": {"correct": 0, "total": 0, "predicted": 0},
        "upscaling": {"correct": 0, "total": 0, "predicted": 0}
    }
}

def format_prompt(prompt):
    """Format prompt for Braintrust"""
    system_prompt = prompt['messages'][0]['content']
    user_prompt = prompt['messages'][1]['content'][0]['text']
    return system_prompt, user_prompt


def parse_verdict(response_content: Any) -> Optional[str]:
    """Parse model verdict with improved error handling and logging."""
    try:
        # Handle case when response_content is already a dict
        if isinstance(response_content, dict):
            return response_content.get("Verdict", "").lower()
        
        # Handle string input with optional ```json delimiters
        if isinstance(response_content, str):
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0]
            elif '{' in response_content and '}' in response_content:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start == -1 or json_end <= json_start:
                    return None
                json_content = response_content[json_start:json_end]
            else:
                json_content = response_content

            # Parse JSON content
            verdict = json.loads(json_content).get("Verdict", "").lower()
            return verdict
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response content: {response_content}")
        return None
    

def encode_image(image_url: str) -> str:
    """Encode image to base64 and detect image format"""
    response = requests.get(image_url)
    image_content = response.content
    
    # Method 1: Check Content-Type header
    content_type = response.headers.get('Content-Type', '')
    if content_type.startswith('image/'):
        image_format = content_type.split('/')[-1]
    else:
        # Method 2: Use imghdr to detect format from binary content
        image_format = imghdr.what(None, h=image_content)
        if not image_format:
            # Default to png if format cannot be detected
            image_format = 'png'
    
    return f"data:image/{image_format};base64,{base64.b64encode(image_content).decode('utf-8')}"


async def get_model_response(client, image_url: str, system_prompt: str, user_prompt: str, temp: float, model_config: float) -> str:
    """Get model prediction for an image with temperature variation"""

    base64_image = encode_image(image_url)

    if model_config['model_name'] == "gemini-1.5-pro":
        role = "model"
    else:
        role = "system"

    response = await client.chat.completions.create(
        model=model_config["model_name"],
        messages=[
            {
                "role": role,
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                            "detail": "low"
                        }
                    },
                ],
            }
        ],
        temperature=temp,
        max_tokens=model_config["max_tokens"]
    )
    print(f"response:{response}")
    return response.choices[0].message.content


def custom_scorer(input, output: Any, expected: Any) -> int:
    try:
        # Convert to standardized format
        y_true = str(expected).lower().strip()
        y_pred = str(parse_verdict(output)).lower().strip()
        
        # Update aggregated results
        aggregated_results["total_samples"] += 1
        if y_true == y_pred:
            aggregated_results["correct_predictions"] += 1
        
        # Update class-specific metrics
        for class_name in ["vectorizing", "upscaling"]:
            if y_true == class_name:
                aggregated_results["class_metrics"][class_name]["total"] += 1
            if y_pred == class_name:
                aggregated_results["class_metrics"][class_name]["predicted"] += 1
            if y_true == class_name and y_pred == class_name:
                aggregated_results["class_metrics"][class_name]["correct"] += 1
        
        # Calculate basic match
        is_correct = y_true == y_pred
        
        # Return multiple scores for different aspects
        return [
            Score(
                name="accuracy",
                score=float(is_correct),
                metadata={
                    "input": input,
                    "output": output,
                    "expected": expected,
                    "prediction": y_pred
                }
            ),
            Score(
                name="class_metrics",
                score=float(is_correct),  # Using accuracy as main score
                metadata={
                    "input": input,
                    "output": output,
                    "expected": expected,
                    "prediction": y_pred,
                    "class_metrics": aggregated_results["class_metrics"],
                }
            ),
            Score(
                name="confidence",
                score=1.0 if is_correct else 0.0,  # Binary confidence for now
                metadata={
                    "input": input,
                    "output": output,
                    "expected": expected,
                    "prediction": y_pred
                }
            )
        ]
    except Exception as e:
        print(f"Error in custom scorer: {str(e)}")
        return [
            Score(
                name="error",
                score=0.0,
                metadata={
                    "error": str(e),
                    "input": input,
                    "output": output,
                    "expected": expected
                }
            )
        ]

def setup_models():
    """Setup all models with proper configuration"""
        
    return {
        "gpt4o": {
            "client": wrap_openai(AsyncOpenAI(
                base_url="https://api.braintrust.dev/v1/proxy",
                default_headers={"x-bt-use-cache": "never"},
                api_key=os.environ["OPENAI_API_KEY"],
                )
            ), 
            "config": {
                "model_name": "gpt-4o",
                "max_tokens": 100
            }
        },
        "gpt4o-mini": {
            "client": wrap_openai(AsyncOpenAI(
                base_url="https://api.braintrust.dev/v1/proxy",
                default_headers={"x-bt-use-cache": "never"},
                api_key=os.environ["OPENAI_API_KEY"],
                )
            ), 
            "config": {
                "model_name": "gpt-4o-mini",
                "max_tokens": 100
            }
        },
        "finetuned-gpt4o": {
            "client": wrap_openai(AsyncOpenAI(
                base_url="https://api.braintrust.dev/v1/proxy",
                default_headers={"x-bt-use-cache": "never"},
                api_key=os.environ["OPENAI_API_KEY_FT"],
                )
            ), 
            "config": {
                "model_name": "ft:gpt-4o-2024-08-06:jiffyshirts-com:vectorisation-upscaling-verdict-v3:AIPhqRj4",
                "max_tokens": 100
            }
        },
        "claude-sonnet": {
            "client": wrap_openai(AsyncOpenAI(
                base_url="https://api.braintrust.dev/v1/proxy",
                default_headers={"x-bt-use-cache": "never"},
                api_key=os.environ["ANTHROPIC_API_KEY"],
                )
            ), 
            "config": {
                "model_name": "claude-3-5-sonnet-latest",
                "max_tokens": 100
            }
        },
        "gemini": {
            "client": wrap_openai(AsyncOpenAI(
                base_url="https://api.braintrust.dev/v1/proxy",
                default_headers={"x-bt-use-cache": "never"},
                api_key=os.environ["GOOGLE_API_KEY"],
                )
            ), 
            "config": {
                "model_name": "gemini-1.5-pro",
                "max_tokens": 100
            }
        }
    }


system_prompt, user_prompt = format_prompt(prompt.build())
models = setup_models()
eval_tasks = []

for temp in temperatures:
    for model_name, model_info in models.items():
        model_client = model_info["client"]
        model_config = model_info["config"]

        experiment_name = f"{model_name}-(temp={temp})-date={datetime.now().strftime('%Y-%m-%d')}"

        async def task_wrapper(data_point, client=model_client, config=model_config):
            """Task wrapper for Braintrust Eval to get model response."""
            try:
                return await get_model_response(
                    client,
                    data_point,
                    system_prompt,
                    user_prompt,
                    temp,
                    config
                )
            except Exception as e:
                print(f"Error in task wrapper: {str(e)}")
                return "error"
            
        eval_task = Eval(
            "Vectorisation vs Upscaling",
            data=experiment.dataset,
            task=task_wrapper,
            scores=[custom_scorer],
            experiment_name=experiment_name,
            metadata={
                "model": model_name,
                "temperature": temp,
                "timestamp": datetime.now().isoformat(),
                "prompt_version": "v1",
                "model_config": {
                "model_name": model_config["model_name"],
                "max_tokens": model_config["max_tokens"],
                "temperature": temp
                            },
                        }
                    )
        eval_tasks.append(eval_task)


async def main():
    for eval_task in eval_tasks:
        await eval_task.run()  # Ensures each Eval instance is individually run for Braintrust compatibility

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
