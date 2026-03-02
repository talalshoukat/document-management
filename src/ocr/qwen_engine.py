"""Qwen OCR engine - strong for Arabic and multi-script documents."""

import torch
from PIL import Image
import unittest
import httpx
import os
import json
import time
import io
from PIL import Image 


def _get_device():
    """Get best available device for Surya models.

    Note: MPS (Apple Silicon) has known issues with Surya causing slow/incorrect results.
    See: https://github.com/pytorch/pytorch/issues/84936
    Only use CUDA if available, otherwise CPU is faster and more reliable.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class QwenEngine:
    """OCR engine using Qwen - ML-based, excellent for Arabic and 90+ languages."""

    def __init__(self):

        # Configuration Constants
        self.APP_URL = os.environ.get("APP_URL")
        self.APP_API_KEY = os.environ.get("APP_API_KEY")
        self.VLM_URL = os.environ.get("VLM_URL")
        self.VLM_API_KEY = os.environ.get("VLM_API_KEY")
        self.VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-30b-a3b-instruct-fp8")
        self.SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() == "true"


    def extract_text(self, image: Image.Image) -> str:
        """Extract full text from image using Surya OCR.

        Uses run_ocr which handles the full pipeline:
        detection (find text regions) → slicing (crop lines) → recognition (read text).
        """
        def docling_main(image):
            PROMPT = """
            You are an expert in extracting text from documents.
            Given an input document extract the text from it while ensuring the structure is maintained.
            If you find an image with schema or some flowchart where text extraction does not make much sense, you can describe the content of the image instead.
            Answer only with the extracted text.
            Extracted text:
            """
            # PROMPT = """
            #     Is there a car in this document? Yes or No 
            #     Answer: 
            # """

            # EXPECTED_TEXT_EXTRACTED = (
            #     "Please find attached the document showcasing my presence at the Hospital today"
            # )


            class TestDoclingConversion():
                def wait_for_service(self, url, timeout=600):
                    print(f"Waiting for {url} to return 200...", flush=True)
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            response = httpx.get(
                                url,
                                headers={"Authorization": f"Bearer {os.environ.get("VLM_API_KEY")}"},
                                timeout=10,
                                verify=False,
                            )
                            if response.status_code == 200:
                                print(f"Successfully connected to {url}", flush=True)
                                return
                        except Exception as e:
                            print(f"Connection attempt failed: {e}", flush=True)
                        time.sleep(2)
                    raise TimeoutError(f"Failed to get 200 from {url} after {timeout} seconds")
                
                def test_conversion_endpoint(self, image):
                    client = httpx.Client(timeout=600.0)

                    endpoint_url = f"{os.environ.get("VLM_URL")}/{os.environ.get("VLM_MODEL")}/v1/chat/completions"
                    models_url = f"{os.environ.get("VLM_URL")}/{os.environ.get("VLM_MODEL")}/v1/models"

                    self.wait_for_service(models_url)

                    vlm_pipeline_model_api = json.dumps(
                        {
                            "url": endpoint_url,
                            "headers": {"Authorization": f"Bearer {os.environ.get("VLM_API_KEY")}"},
                            "params": {"model": os.environ.get("VLM_MODEL"), "max_tokens":4096},
                            "timeout": 600,
                            "concurrency": 1,
                            "prompt": PROMPT,
                            "response_format": "markdown",
                            "temperature": 0.6,
                        }
                    )

                    parameters = {
                        "from_formats": ["docx", "pptx", "html", "image", "pdf", "asciidoc", "md", "xlsx"],
                        "to_formats": ["md", "json", "html", "text", "doctags"],
                        "image_export_mode": "placeholder",
                        "do_ocr": True,
                        "ocr_engine": "rapidocr",
                        "ocr_lang": ["eng", "ara"],
                        "force_ocr": False,
                        "pdf_backend": "dlparse_v2",
                        "table_mode": "fast",
                        "abort_on_error": False,
                        "pipeline": "vlm",
                        "vlm_pipeline_model_api": vlm_pipeline_model_api
                    }
                    img_byte_arr = io.BytesIO()
                    # image.save(img_byte_arr, format='PNG')  # Save it as PNG or any other format
                    image.save(img_byte_arr, format='PDF')
                    img_byte_arr.seek(0)  # Rewind the byte stream to the beginning

                    # # Adjust path to find data/Doc1.pdf from test/test.py
                    # current_dir = os.path.dirname(__file__)
                    # fname = "image.pdf"
                    # # Go up one level to root, then into data
                    # file_path = os.path.join(current_dir, "..", "Data", fname)
                    # file_path = os.path.abspath(file_path)

                    # extension = fname.split(".")[-1]

                    # files = {
                    #     "files": (fname, open(file_path, "rb"), f"application/{extension}"),
                    # }
                    # Prepare the files object with the image data
                    files = {
                        "files": ("image.png", img_byte_arr, "image/png"),  # Adjust the MIME type as necessary
                    }

                    headers = {"X-Api-Key": os.environ.get("APP_API_KEY")}
                    convert_url = f"{os.environ.get("APP_URL")}/v1/convert/file"
                    print(f"Querying the endpoint: {convert_url}", flush=True)
                    response = client.post(convert_url, files=files, data=parameters, headers=headers)
                    print(f"Got answer from the endpoint: {convert_url}", flush=True)
                    print(response.json())

                    data = response.json()

                    # assert output
                    print(data)
                    # html = data["document"]["html_content"]
                    html = data["document"]["md_content"]
                    print(f"Passed test_conversion_endpoint")
                    return html

            
            docling = TestDoclingConversion()
            data = docling.test_conversion_endpoint(image)

            return data
        data = docling_main(image)
        return data

    def extract_with_confidence(self, image: Image.Image) -> list[dict]:
        """Extract text with bounding boxes and confidence."""
        text = self.extract_text(image)
        results = []
        for line in text.split("\n"):
            if line.strip():
                results.append({
                    "text": line.strip(),
                    "bbox": None,
                    "confidence": 1.0,
                })
        return results
