# %%
import base64
import json
import logging
import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from loguru import logger
from pydantic import ValidationError

from src.schemas.pathology import PathologyExtractionResult

assert load_dotenv()

logging.basicConfig(level=logging.INFO)

os.environ["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"]
os.environ["AWS_REGION"]
os.environ["AWS_PROFILE"]

# %%

# Initialize Langfuse
langfuse = get_client()
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

langfuse_handler = CallbackHandler()

# %%

logger.add(
    "logs/vision_pathology.log",
    rotation="10 MB",
    retention=5,
    format="{time} {level} {message}",
)

REPORT_PATH = Path("../data/synthetic_pathology_report_AML.pdf").resolve()
SCHEMA_PATH = Path("./src/schemas/pathology.py").resolve()
OUTPUT_FILE = "pathology_results.json"

PROMPT = """\
You are a clinical data extraction assistant. You have been given images of each page of a pathology report PDF.

Extract the following fields from the report and return them as JSON matching the schema below:

**Schema** (from `{schema_path}`):
```python
{schema_source}
```

**Fields to extract:**
- **age**: The patient's age (integer as string), if present.
- **primary_diagnosis**: The primary diagnosis from the report.
- **performance_status**: The ECOG or similar performance status, if present.

**Citation rules — every extracted value MUST include a verifiable citation:**
- Each field is a `CitedField` with a `value` and a `citation`.
- A `citation` contains `page` (1-indexed page number) and `text` (the **exact** quote from the PDF that supports the value — copy it verbatim, do not paraphrase).
- If a field is not present in the report, set it to `null` rather than fabricating a citation.

**Output:** Return ONLY valid JSON matching the `PathologyExtractionResult` schema. No prose, no markdown fences.
"""


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[tuple[int, bytes]]:
    """Convert each page of a PDF to a PNG image. Returns list of (page_num, png_bytes)."""
    doc = pymupdf.open(pdf_path)
    pages = []
    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=matrix)
        pages.append((page_num + 1, pix.tobytes("png")))
    doc.close()
    logger.info("Converted {} pages to images from {}", len(pages), pdf_path.name)
    return pages


def build_message(pages: list[tuple[int, bytes]], schema_source: str) -> HumanMessage:
    """Build a multimodal HumanMessage with page images and the extraction prompt."""
    content: list[dict] = []

    for page_num, png_bytes in pages:
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")
        content.append({"type": "text", "text": f"--- Page {page_num} ---"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    content.append(
        {
            "type": "text",
            "text": PROMPT.format(
                schema_path=SCHEMA_PATH.name,
                schema_source=schema_source,
            ),
        }
    )

    return HumanMessage(content=content)


# %%


def run(report_path: Path = REPORT_PATH) -> PathologyExtractionResult:
    logger.info("Starting vision extraction for {}", report_path)

    # Convert PDF to images
    pages = pdf_to_images(report_path)

    # Read schema source for the prompt
    schema_source = SCHEMA_PATH.read_text()

    # Build the multimodal message
    message = build_message(pages, schema_source)

    # Invoke the LLM
    llm = ChatBedrock(
        model="us.anthropic.claude-sonnet-4-6",
        region=os.environ["AWS_REGION"],
        max_tokens=4096,
    )

    logger.info("Invoking LLM with {} page images...", len(pages))
    response = llm.invoke([message], config={"callbacks": [langfuse_handler]})

    raw_text = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("LLM response (first 500 chars): {}", raw_text[:500])

    # Parse and validate with Pydantic
    data = json.loads(raw_text)
    try:
        result = PathologyExtractionResult.model_validate(data)
    except ValidationError as e:
        print(e.errors())
        return
    logger.info("Pydantic validation passed")

    # Write output
    output_path = Path(OUTPUT_FILE)
    output_path.write_text(json.dumps(result.model_dump(), indent=2))
    logger.info("Results written to {}", output_path.resolve())

    return result


# %%

result = run()
print(json.dumps(result.model_dump(), indent=2))
