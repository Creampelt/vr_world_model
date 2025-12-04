import requests
import json
import logging

# #!/bin/bash

# API_URL="http://0.0.0.0:8000/generate"
# PROMPTS_JSON='{
#   "positive_prompts": ["a video of the tree being covered by a sandstorm", "a video of the tree in winter"],
#   "negative_prompts": ["blurry, low quality", "deformed, distorted" ]
# }'
# IMAGES="./tree.png"

# CONNECT_TIMEOUT=10

# curl \
#     --connect-timeout $CONNECT_TIMEOUT \
#     -X POST "$API_URL" \
#     -F "prompts_json=$PROMPTS_JSON" \
#     -F "images=@$IMAGES" \
#     -F "images=@$IMAGES" \


if __name__ == "__main__":
    API_URL = "http://0.0.0.0:8000/generate"

    prompts_json = {
        "positive_prompts": [
            "a video of the tree being covered by a sandstorm",
            "a video of the tree in winter",
        ],
        "negative_prompts": ["blurry, low quality", "deformed, distorted"],
    }
    
    files = [("prompts_json", (None, json.dumps(prompts_json)))]
    files += [("images", open("./tree.png", "rb")), ("images", open("./tree.png", "rb"))]

    response = requests.post(API_URL, files=files)
    logging.info(response.text)
    
    print(response.text)
