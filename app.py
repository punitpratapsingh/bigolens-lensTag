import os
import re
import json
import base64
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Load .env for local development
load_dotenv()

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set in environment variables!")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize model
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="LensTag - Auto Tagging API")

# ------------------------------------------------------
# NEW STRICT JSON PROMPT
# ------------------------------------------------------
PROMPT = """
You are LensTag, an advanced AI that extracts eCommerce tags & attributes.

Output MUST follow exactly this JSON structure:

{
  "category": "",
  "sub_category": "",
  "attributes": {
    "color": "",
    "material": "",
    "pattern": "",
    "style": "",
    "gender": "",
    "occasion": "",
    "segment": "",
    "metal_type": "",
    "closing_type": "",
    "shape": "",
    "craft": ""
  },
  "auto_tags": [],
  "seo_description": ""
}

RULES:
- Return ONLY pure JSON.
- Never use ```json or ``` or markdown.
- Never include explanations.
- auto_tags must contain 10–15 lowercase tags.
- Never return null — use empty strings instead.
- If unsure, make the best guess.

Now analyze the image and respond with clean JSON only:
"""


# ------------------------------------------------------
# LensTag Endpoint
# ------------------------------------------------------
@app.post("/lensTag")
async def lens_tag(image: UploadFile = File(...)):
    # Read image
    image_bytes = await image.read()
    img_b64 = base64.b64encode(image_bytes).decode()

    # Get response from Gemini
    gemini_output = model.generate_content(
        [
            PROMPT,
            {"mime_type": image.content_type, "data": img_b64}
        ]
    )

    raw_text = gemini_output.text

    # Remove markdown fences if present
    clean_text = re.sub(r"```json|```", "", raw_text).strip()

    # Try to parse JSON
    try:
        parsed_json = json.loads(clean_text)
        return JSONResponse(content=parsed_json, status_code=200)

    except Exception:
        # Parsing failed → return debugging info
        return JSONResponse(
            status_code=200,
            content={
                "error": "JSON parsing failed. Check model output.",
                "raw_output": raw_text
            }
        )


# ------------------------------------------------------
# Health Endpoint
# ------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "OK 🚀", "service": "LensTag Gemini Flash 2.5"}
