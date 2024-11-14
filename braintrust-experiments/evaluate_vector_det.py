import os
import yaml
import json
import random
from datetime import datetime
from braintrust import Eval
from braintrust.wrappers.langchain import BraintrustTracer
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

tracer = BraintrustTracer()

def parse_verdict(response_content):
    """Parse verdict from response content handling all formats"""
    try:
        # First try GPT format (with ```json```)
        verdict = json.loads(response_content.split('```json')[1].split('```')[0])['Verdict'].lower()
    except (IndexError, json.JSONDecodeError):
        try:
            # Try to extract just the JSON part from Claude's verbose response
            # Find the first occurrence of a JSON-like structure
            json_start = response_content.find('{')
            json_end = response_content.find('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start:json_end]
                verdict = json.loads(json_str)['Verdict'].lower()
            else:
                raise json.JSONDecodeError("No JSON found", response_content, 0)
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            print(f"Response content: {response_content}")
            return None
    return verdict

def get_model_response(client, image_url):
    """Task function that generates model predictions using client.invoke"""
    try:
        # Load prompts
        with open('/Users/vatsaljha/Downloads/scripts/dtf_qa/gpt_finetune/prompt_vec_up.yaml', "r") as file:
            prompts_dict = yaml.safe_load(file)

        messages = [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": prompts_dict['P1'].strip()
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "low"
                    }
                }
            ])
        ]
        
        # Get response from model
        response = client.invoke(messages)
        
        # Parse verdict using the new helper function
        verdict = parse_verdict(response.content)
        if verdict is None:
            return "error"
            
        return verdict
        
    except Exception as e:
        print(f"Error in get_model_response: {str(e)}")
        return "error"

def exact_match(input, expected, output):
    return 1 if output == expected else 0

def extract_image_url(message):
    """Extract image URL from message content"""
    for content in message['content']:
        if isinstance(content, dict) and content['type'] == 'image_url':
            return content['image_url']['url']
    return None

def extract_ground_truth(message):
    """Extract ground truth verdict from message content"""
    content = message['content']
    verdict = json.loads(content.split('```json')[1].split('```')[0])['Verdict']
    return verdict.lower()

# Define clients
clients = [
    ChatOpenAI(
        base_url="https://api.braintrust.dev/v1/proxy",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        temperature=0,
        callbacks=[tracer],
        max_tokens = 100
    ),
    ChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='ft:gpt-4o-2024-08-06:jiffyshirts-com:vectorisation-upscaling-verdict-v3:AIPhqRj4',
        temperature=0,
        callbacks=[tracer],
        max_tokens = 100
    ),
    ChatOpenAI(
        base_url="https://api.braintrust.dev/v1/proxy",  # Add Braintrust proxy
        openai_api_key=os.getenv('ANTHROPIC_API_KEY'),  # Can use Anthropic key directly
        model='claude-3-5-sonnet-latest',
        temperature=0,
        callbacks=[tracer],
        max_tokens = 100
    ),
    ChatOpenAI(
        base_url="https://api.braintrust.dev/v1/proxy",
        openai_api_key=os.getenv('GOOGLE_API_KEY'),
        model='gemini-1.5-pro',
        temperature=0,
        callbacks=[tracer],
        max_tokens = 100
    )
]

# Load validation data
validation_file = "/Users/vatsaljha/Downloads/scripts/dtf_qa/gpt_finetune/data/validation_data_v3.jsonl"
with open(validation_file, 'r') as f:
    all_data = [json.loads(line) for line in f]
    # Format data for Braintrust Eval
    eval_data = [
        {
            "input": {
                "image_url": extract_image_url(item['messages'][2])
            },
            "expected": extract_ground_truth(item['messages'][3]),
            "metadata": {
                "idx": i,
            }
        }
        for i, item in enumerate(random.sample(all_data, 200))  # Randomly select 10 samples
    ]

# Run evaluation for each model sequentially
for client in clients:
    Eval(
        "Vectorize Image Detection",
        data=eval_data,
        task=lambda x: get_model_response(client, x["image_url"]),
        scores=[exact_match],
        experiment_name=f"Verdict Classification - {client.model_name}",
        metadata={
            "model": client.model_name,
            "timestamp": datetime.now().isoformat()
        },
        # max_concurrency=1  # Run sequentially to avoid threading issues
    )
