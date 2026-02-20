"""
01 - Multi-format Document Loader
PDF, Image, Text, Word, CSV, Excel, PowerPoint, Markdown ဖိုင်တွေကို
Load လုပ်ပြီး LangChain Document objects အဖြစ် ပြောင်းပေးတယ်။

Dependencies:
    pip install pytesseract Pillow python-docx openpyxl
    sudo apt install tesseract-ocr
"""

import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

# Extension → loader mapping
_LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".csv":  CSVLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc":  UnstructuredWordDocumentLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt":  UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls":  UnstructuredExcelLoader,
}


def load_file(file_path: str) -> list:
    """
    ဖိုင် extension ကို ကြည့်ပြီး သင့်တော်တဲ့ loader နဲ့ load လုပ်ပေးတယ်။

    Supported: .pdf, .txt, .csv, .md, .docx, .doc, .pptx, .ppt, .xlsx, .xls,
               .png, .jpg, .jpeg, .bmp, .tiff, .tif, .webp (OCR)

    Args:
        file_path: ဖိုင်ရဲ့ path

    Returns:
        list: LangChain Document objects များ

    Raises:
        ValueError: support မလုပ်တဲ့ file type ဖြစ်ရင်
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Image → OCR
    if ext in IMAGE_EXTENSIONS:
        return _load_image(file_path)

    # Known document types
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'\n"
            f"Supported: {', '.join(sorted(set(list(_LOADER_MAP.keys()) + list(IMAGE_EXTENSIONS))))}"
        )

    loader = loader_cls(file_path)
    return loader.load()


def _load_image(file_path: str) -> list:
    """
    Image ဖိုင်ကို OCR (pytesseract) သုံးပြီး text extract လုပ်ကာ
    Document object အဖြစ် return ပြန်ပေးတယ်။
    """
    from PIL import Image
    import pytesseract

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    if not text.strip():
        return []

    doc = Document(
        page_content=text,
        metadata={"source": file_path, "type": "image"},
    )
    return [doc]
