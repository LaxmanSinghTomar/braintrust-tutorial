#################################################################################################
# This script aims to capture both LLM and Non-LLM outputs, as well as user feedback in BrainTrust
#################################################################################################


# Importing Libraries
import os
import asyncio
import aiofiles
import aiohttp
import base64
import json
import re
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# BrainTrust
from braintrust import init_logger, traced, wrap_openai
from openai import AsyncOpenAI

logger = init_logger(project="Tracing Non-LLM with User Feedback")
 

#client = wrap_openai(AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]))

# Load Environment Vars
load_dotenv(".env")

client = wrap_openai(AsyncOpenAI(
  base_url="https://api.braintrust.dev/v1/proxy",
  default_headers={"x-bt-use-cache": "never"},
  api_key=os.environ["OPENAI_API_KEY"], # Braintrust Key works when Vendors are configured in Settings, or directly for another LLM like Anthropic, etc. keys here too
))

# Constants
IMAGE_DESCRIPTION_PROMPT = """You are an expert image analyst with a keen eye for detail. Your task is to provide a comprehensive description of an image, clearly distinguishing between the main subject(s) and the background elements. This description will be used to assess the quality of background removal in a subsequent step. Please follow these guidelines:

1. Overall Image Description:
   - Provide a brief overview of what the image depicts.
   - Mention the type of image (e.g., photograph, illustration, logo, graphic design, etc.).

2. Main Subject Identification:
   - Clearly identify and describe the main subject(s) of the image.
   - IMPORTANT: In graphic designs, logos, and similar images, consider text, design elements, and visual effects (like splashes, gradients, or patterns) as part of the main subject.
   - Include details such as:
     - What or who the subject is (including text and design elements)
     - Its position within the image
     - Notable features or characteristics
     - Colors and textures associated with the subject

3. Background Description:
   - Describe the background in detail, including:
     - Colors and patterns
     - Any identifiable objects or elements
     - Texture or consistency (e.g., solid color, gradient, textured)
   - IMPORTANT: For graphic designs and logos, the background is often a uniform color or simple pattern. Complex visual elements are typically part of the main subject.

4. Challenging Areas:
   - Identify areas that might be challenging for background removal, such as:
     - Fine details (e.g., hair, fur, intricate patterns)
     - Transparent or translucent elements
     - Areas where the subject and background have similar colors or textures
     - Complex edges or boundaries between the subject and background

5. Element Classification:
   - Clearly state which elements should be considered part of the subject (to be preserved) and which are part of the background (to be removed).
   - Remember: In graphic designs and logos, most visual elements, including text and design features, are typically part of the subject.
   - If there are any ambiguous elements, explain why they might be challenging to classify.

Please provide your analysis in the following JSON format:

```json
{{
  "image_type": "Type of image (e.g., photograph, logo, illustration, graphic design)",
  "overall_description": "Brief overview of the entire image",
  "main_subject": {{
    "description": "Detailed description of the main subject(s), including text and design elements",
    "key_features": ["List of notable features"],
    "colors": ["List of primary colors associated with the subject"],
    "position": "Description of subject's position in the image"
  }},
  "background": {{
    "description": "Detailed description of the background",
    "colors": ["List of primary background colors"],
    "elements": ["List of any identifiable background elements"],
    "texture": "Description of background texture or consistency"
  }},
  "challenging_areas": [
    {{
      "location": "Description of the challenging area's location",
      "reason": "Explanation of why this area is challenging for background removal"
    }}
  ],
  "element_classification": {{
    "subject_elements": ["List of elements that should be preserved"],
    "background_elements": ["List of elements that should be removed"],
    "ambiguous_elements": [
      {{
        "element": "Description of the ambiguous element",
        "explanation": "Why this element is difficult to classify"
      }}
    ]
  }}
}}
```

Please ensure your entire response can be parsed as a valid JSON object. Your description should be detailed and precise, providing a clear understanding of what constitutes the subject and background in the image. This information will be crucial for accurately assessing background removal quality in the next step."""


DETECT_NO_BG_REMOVAL_PROMPT = """You are an expert image analyst specializing in background removal techniques. You will be presented with three pieces of information:
1. A detailed description of the original image (provided below)
2. The original image itself
3. A background-removed version of the image

Your task is to analyze these inputs and identify any cases where no meaningful removal has occurred. Use the detailed image description to inform your analysis.

Detailed Image Description:
{IMAGE_DESCRIPTION}

Please follow these steps in your analysis:

1. Review the Detailed Image Description:
   - Understand the main subject(s) and background elements as described.
   - Note the challenging areas identified for background removal.
   - Pay attention to the element classification provided.

2. Original Image Analysis:
   - Compare the original image with the provided description to confirm its accuracy.
   - Note any discrepancies or additional observations.

3. Background-Removed Image Analysis:
   - Compare the background-removed image to both the original image and the detailed description.
   - Determine if meaningful background removal has occurred, considering the described background elements.
   - Look for the following significant issues:

     a) No Meaningful Removal:
     - Determine if the background elements described have been removed or significantly altered.
     - Be especially careful with images described as having uniform backgrounds.

Please provide your analysis in the following JSON format:

```json
{{
  "background_removed_image_analysis": {{
    "removal_occurred": true,
    "explanation": "Brief explanation of removal assessment, referencing the image description"
  }},
  "significant_issues": {{
    "no_removal": {{
      "detected": false,
      "explanation": "Explain if no meaningful removal occurred, referencing the image description"
    }}
  }},
}}
```

Notes:
- Use the detailed image description as a reference point throughout your analysis.
- If removal_occurred is false, then no_removal is expected to be true indicating that no background removal occurred and the new image is same as original.

Please ensure your entire response can be parsed as a valid JSON object. Your analysis should focus on major issues that have a substantial impact on the image quality or subject integrity, and the overall evaluation should accurately reflect these findings, all while leveraging the context provided by the detailed image description."""

@traced
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@traced
async def identify_image_description_openai(original_image_path, model_name="gpt-4o"):
    """
    Identifies the image description using OpenAI.
    """
    try:
        print("Starting image encoding...")
        original_base64_image = encode_image(original_image_path)
        print("Image encoded successfully")

        print("Making API request...")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": IMAGE_DESCRIPTION_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{original_base64_image}",
                                "detail": "high"
                            }
                        },
                    ]
                }
            ],
            temperature=0,
            max_tokens=1000
        )
        print("API response received")
        print(f"Response type: {type(response)}")
        print(f"Response content: {response.choices[0].message.content[:100]}...")  # First 100 chars
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in identify_image_description_openai: {str(e)}")
        print(f"Error type: {type(e)}")


@traced
async def detect_bg_removal_openai(original_image_path, bg_removed_image_path, input_prompt, model_name="gpt-4o"):
    """
    Detects if background removal has occurred or not.
    """
    original_base64_image = encode_image(original_image_path)
    bg_removed_base64_image = encode_image(bg_removed_image_path)

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{original_base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{bg_removed_base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1000
    )
    return response.choices[0].message.content


@traced
def add_checkered_background(input_path, images_dir, checker_size=20, to_save=True):
    """
    Adds a checkered background to an image.
    """
    # Open the original image
    original = Image.open(input_path).convert("RGBA")

    # Create a new image with white background
    result = Image.new("RGBA", original.size, (255, 255, 255, 255))

    # Create a drawing object
    draw = ImageDraw.Draw(result)

    # Draw the checkered pattern
    for i in range(0, result.width, checker_size):
        for j in range(0, result.height, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                draw.rectangle([i, j, i + checker_size, j + checker_size], fill=(200, 200, 200))

    # Attempt to paste the original image onto the checkered background
    try:
        result.paste(original, (0, 0), original)
    except ValueError as e:
        raise ValueError(f"Error processing {input_path}: {str(e)}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    output_directory = f"{images_dir}/checkered_bg_images/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, f"{base_name}.png")

    # Save the result
    if to_save:
        result.save(output_path)

    return output_path

@traced
def clean_and_parse_json(input_string):
    cleaned_string = re.sub(r"```json\s*|\s*```", "", input_string)
    cleaned_string = cleaned_string.strip()

    try:
        parsed_dict = json.loads(cleaned_string)
        return parsed_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return input_string

@traced
def try_parse_json(value):
    try:
        return clean_and_parse_json(value)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return value

@traced
async def bg_removal_detector_driver(original_image_path, bg_removed_image_path):
    try:
        # identify image description
        image_description = await identify_image_description_openai(original_image_path)
        print(f"image_description: {image_description}")
        
        # check if bg-removed has occurred
        print(f"Starting bg removal detection:")
        model_response = await detect_bg_removal_openai(original_image_path=original_image_path, bg_removed_image_path=bg_removed_image_path, input_prompt=DETECT_NO_BG_REMOVAL_PROMPT.format(IMAGE_DESCRIPTION=image_description))
        print(f"model_response: {model_response}")
    
        return try_parse_json(model_response)
    except Exception as e:
        print(f"Error in bg_removal_detector_driver: {str(e)}")
        return ""

@traced
async def download_image(session, url, filename):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                f = await aiofiles.open(filename, mode='wb')
                await f.write(await response.read())
                await f.close()
                print(f"Downloaded {filename}")
                return filename
            else:
                print(f"Failed to download {url}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

@traced
def process_images(batch, identifier_column, images_dir):
    results = []
    for _, row in batch.iterrows():
        identifier = row[identifier_column]
        analyses = {
            'bg_removal_analysis': None,
        }
        
        orig_filename = f"{images_dir}/{identifier}_original.png"
        final_filename = f"{images_dir}/{identifier}_bg_removed.png"
        if os.path.exists(orig_filename) and os.path.exists(final_filename):
            analyses['bg_removal_analysis'] = bg_removal_detector_driver(orig_filename, final_filename)
        
        results.append({
            identifier_column: identifier,
            'bg_removal_analysis': analyses['bg_removal_analysis'],
        })
    
    return results

@traced
async def driver(original_image_url, bg_removed_image_url):
    original_filename = "original.png"
    bg_removed_filename = "bg_removed.png"
    
    downloaded_files = []
    
    try:
        # Create an aiohttp session
        async with aiohttp.ClientSession() as session:
            # Download images
            original_image_path = await download_image(session, original_image_url, original_filename)
            bg_removed_image_path = await download_image(session, bg_removed_image_url, bg_removed_filename)
            
            if original_image_path:
                downloaded_files.append(original_image_path)
            if bg_removed_image_path:
                downloaded_files.append(bg_removed_image_path)
        
        if original_image_path and bg_removed_image_path:
            # Create a checkered-background image if possible
            checkered_bg_path = add_checkered_background(bg_removed_image_path, ".")
            if checkered_bg_path:
                downloaded_files.append(checkered_bg_path)
            
            # Detect bg-removal
            bg_analysis = await bg_removal_detector_driver(original_image_path, checkered_bg_path)
            removal_analysis = {
                "result": "success",
                "verdict": bg_analysis['background_removed_image_analysis']['removal_occurred'],
                "explanation": bg_analysis['background_removed_image_analysis']['explanation'],
            }
            return removal_analysis
        else:
            return {
                "result": "failure",
                "verdict": False,
                "explanation": "Failed to download one or both images",
            }
    
    except Exception as e:
        return {
            "result": "failure",
            "verdict": False,
            "explanation": f"An error occurred: {str(e)}",
        }
    
    finally:
        # Clean up downloaded files
        for file in downloaded_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted file: {file}")
                except Exception as e:
                    print(f"Error deleting file {file}: {str(e)}")


if __name__ == "__main__":

    # # NO CHANGE
    # no_change_original = "https://jiffy-transfers.imgix.net/2/attachments/8rd1z57ekkg9q22ebaabqaxvuxmi?ixlib=rb-0.3.5"
    # no_change_bg_removed = "https://jiffy-transfers.imgix.net/2/attachments/b91m0lvbtj5fwc94k2vjd2x4nezr?ixlib=rb-0.3.5"

    # output = asyncio.run(driver(no_change_original, no_change_bg_removed))
    # print(output)

    # CHANGE
    change_original = "https://jiffy-transfers.imgix.net/2/attachments/b8pfg6aty48imr0hrbhfkhyrh6gk?ixlib=rb-0.3.5"
    change_bg_removed = "https://jiffy-transfers.imgix.net/2/attachments/n65lj6rngnc4gyfioupb8guta16f?ixlib=rb-0.3.5"

    # Uncomment following for capturing single user feedback

    # with logger.start_span() as span:
    #     body = {
    #         "user_id": "989",
    #         "original_image_url": change_original,
    #         "bg_removed_image_url": change_bg_removed,
    #     }
    #     result = asyncio.run(driver(change_original, change_bg_removed))
    #     span.log(input=body, output=result)
    #     print(result)
    #     body["request_id"] = span.id
    

    # comment = "Testing User Feedback from SDK; seems all good."
    
    # scores = {
    #     "correctness": 0,
    # }

    # expected_val = 1

    # logger.log_feedback(
    #     id=body['request_id'],
    #     scores=scores,
    #     expected=expected_val,
    #     comment=comment,
    #     metadata={
    #         "user_id": body['user_id'],
    #     },
    # )

    # For Multiple Users
    with logger.start_span() as span:
        body = {
            "user_id": "989",
            "original_image_url": change_original,
            "bg_removed_image_url": change_bg_removed,
        }
        result = asyncio.run(driver(change_original, change_bg_removed))
        print(result)
        span.log(input=body, output=result)
        body["request_id"] = span.export()

    
    with logger.start_span("Feedback", parent=body['request_id']) as span:
        logger.log_feedback(
            id=span.id,  # Use the newly created span's id, instead of the original request's id
            scores={
                "correctness": 0,
            },
            expected=1,
            comment="Testing User Feedback from SDK; feedback from multiple users; from #1.",
            metadata={
                "user_id": body['user_id'],
            },
        )

    
    with logger.start_span("Feedback", parent=body['request_id']) as span:
        logger.log_feedback(
            id=span.id,  # Use the newly created span's id, instead of the original request's id
            scores={
                "correctness": 1,
            },
            expected=1,
            comment="Testing User Feedback from SDK; feedback from multiple users; from #2.",
            metadata={
                "user_id": "1000",
            },
        )

    with logger.start_span("Feedback", parent=body['request_id']) as span:
        logger.log_feedback(
            id=span.id,  # Use the newly created span's id, instead of the original request's id
            scores={
                "correctness": 1,
            },
            expected=1,
            comment="Testing User Feedback from SDK; feedback from multiple users; from #3.",
            metadata={
                "user_id": 1100,
            },
        )
