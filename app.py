import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="LensTag API", version="1.0")

# ---------------------------
# HEALTH CHECK API
# ---------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LensTag API", "version": "1.0"}


def image_to_base64(image_file):
    return base64.b64encode(image_file).decode("utf-8")


def build_prompt():
    return """
You are LensTag, an advanced AI that extracts eCommerce tags & attributes.

Return JSON only in the EXACT format below:

{
  "tags": ["tag1", "tag2", "..."],
  "category": "main_category",
  "attributes": {
    "color": "",
    "pattern": "",
    "material": "",
    "sleeve": "",
    "neck": "",
    "fit": "",
    "style": "",
    "occasion": ""
  }
}

Rules:
- Do NOT include explanations.
- Category must be a single high-level fashion category (e.g., tshirt, shirt, saree, kurta, jeans, footwear, handbag, dress, top, jacket, shorts).
- Keep tags short (5–15).
- If unsure of an attribute, return empty string.
"""
    

@app.post("/lensTag")
async def get_tags_and_attributes(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        b64_img = image_to_base64(img_bytes)

        response = client.chat.completions.create(
            model="gpt-4o-mini",   # ultra fast + cheap + reliable
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}"
                            }
                        }
                    ]
                }
            ]
        )

        output = response.choices[0].message["content"]

        return JSONResponse(content={"result": output})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
