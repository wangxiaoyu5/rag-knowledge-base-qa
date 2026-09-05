"""
文档加载模块 - 支持多种文档格式的解析与加载
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """文档加载器基类"""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document 列表
        """
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """
        检查是否支持该文件类型
        
        Args:
            file_extension: 文件扩展名（如 .pdf, .docx）
            
        Returns:
            是否支持
        """
        pass


class PDFLoader(BaseDocumentLoader):
    """PDF 文档加载器"""

    def load(self, file_path: str) -> List[Document]:
        """加载 PDF 文档"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # 添加文件元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'pdf',
                    'file_name': Path(file_path).name
                })

            logger.info(f"成功加载 PDF 文件: {file_path}, 共 {len(documents)} 页")
            return documents

        except Exception as e:
            logger.error(f"加载 PDF 文件失败: {file_path}, 错误: {str(e)}")
            raise

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.pdf'


class DocxLoader(BaseDocumentLoader):
    """Word 文档加载器"""

    def load(self, file_path: str) -> List[Document]:
        """加载 Word 文档"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

            # 添加文件元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'docx',
                    'file_name': Path(file_path).name
                })

            logger.info(f"成功加载 Word 文件: {file_path}")
            return documents

        except Exception as e:
            logger.error(f"加载 Word 文件失败: {file_path}, 错误: {str(e)}")
            raise

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.docx', '.doc']


class TextLoader(BaseDocumentLoader):
    """文本文件加载器（支持 .txt, .md, .py 等）"""

    def load(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        try:
            from langchain_community.document_loaders import TextLoader as LangChainTextLoader
            loader = LangChainTextLoader(file_path, encoding='utf-8')
            documents = loader.load()

            # 添加文件元数据
            file_ext = Path(file_path).suffix.lower()
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': file_ext.lstrip('.'),
                    'file_name': Path(file_path).name
                })

            logger.info(f"成功加载文本文件: {file_path}")
            return documents

        except UnicodeDecodeError:
            # 尝试使用其他编码
            try:
                loader = LangChainTextLoader(file_path, encoding='gbk')
                documents = loader.load()
                file_ext = Path(file_path).suffix.lower()
                for doc in documents:
                    doc.metadata.update({
                        'source': file_path,
                        'file_type': file_ext.lstrip('.'),
                        'file_name': Path(file_path).name
                    })
                return documents
            except Exception as e:
                logger.error(f"加载文本文件失败: {file_path}, 错误: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"加载文本文件失败: {file_path}, 错误: {str(e)}")
            raise

    def supports(self, file_extension: str) -> bool:
        text_extensions = ['.txt', '.md', '.markdown', '.py', '.java', '.cpp',
                          '.c', '.h', '.js', '.ts', '.html', '.css', '.json',
                          '.xml', '.yaml', '.yml', '.ini', '.conf', '.sh', '.bat']
        return file_extension.lower() in text_extensions


class ImageProcessor:
    """图片处理器 - 提取图片中的文字和内容描述"""

    def __init__(self, use_ocr: bool = True, use_multimodal: bool = False):
        """
        初始化图片处理器
        
        Args:
            use_ocr: 是否使用OCR提取文字
            use_multimodal: 是否使用多模态模型生成图片描述
        """
        self.use_ocr = use_ocr
        self.use_multimodal = use_multimodal
        self._ocr_engine = None
        self._multimodal_model = None

    def _load_ocr_engine(self):
        """懒加载OCR引擎"""
        if self._ocr_engine is None and self.use_ocr:
            try:
                import pytesseract
                from PIL import Image
                self._ocr_engine = {
                    'pytesseract': pytesseract,
                    'Image': Image
                }
                logger.info("OCR引擎加载完成")
            except ImportError:
                logger.warning("未安装 pytesseract 或 PIL，OCR功能不可用")
                self.use_ocr = False
        return self._ocr_engine

    def _load_multimodal_model(self):
        """懒加载多模态模型"""
        if self._multimodal_model is None and self.use_multimodal:
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self._multimodal_model = {
                    'processor': processor,
                    'model': model
                }
                logger.info("多模态模型加载完成")
            except ImportError:
                logger.warning("未安装 transformers，多模态功能不可用")
                self.use_multimodal = False
        return self._multimodal_model

    def extract_text_from_image(self, image_path: str) -> str:
        """
        从图片中提取文字（OCR）
        
        Args:
            image_path: 图片路径
            
        Returns:
            提取的文字
        """
        ocr_engine = self._load_ocr_engine()
        if not ocr_engine:
            return ""

        try:
            image = ocr_engine['Image'].open(image_path)
            text = ocr_engine['pytesseract'].image_to_string(image, lang='chi_sim+eng')
            return text.strip()
        except Exception as e:
            logger.error(f"OCR识别失败: {image_path}, 错误: {str(e)}")
            return ""

    def generate_image_description(self, image_path: str) -> str:
        """
        生成图片内容描述（多模态）
        
        Args:
            image_path: 图片路径
            
        Returns:
            图片描述文本
        """
        multimodal_model = self._load_multimodal_model()
        if not multimodal_model:
            return ""

        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            inputs = multimodal_model['processor'](image, return_tensors="pt")
            out = multimodal_model['model'].generate(**inputs)
            description = multimodal_model['processor'].decode(out[0], skip_special_tokens=True)
            return description.strip()
        except Exception as e:
            logger.error(f"多模态描述生成失败: {image_path}, 错误: {str(e)}")
            return ""

    def process_image(self, image_path: str) -> str:
        """
        完整处理图片，返回综合文本描述
        
        Args:
            image_path: 图片路径
            
        Returns:
            图片的综合文本描述（OCR文字 + 图片描述）
        """
        parts = []

        # OCR提取文字
        if self.use_ocr:
            ocr_text = self.extract_text_from_image(image_path)
            if ocr_text:
                parts.append(f"【图片文字内容】\n{ocr_text}")

        # 多模态描述
        if self.use_multimodal:
            description = self.generate_image_description(image_path)
            if description:
                parts.append(f"【图片内容描述】\n{description}")

        # 如果都失败，返回文件名作为描述
        if not parts:
            parts.append(f"【图片文件】{Path(image_path).name}")

        return "\n\n".join(parts)


class ImageLoader(BaseDocumentLoader):
    """图片文件加载器"""

    def __init__(self):
        self.image_processor = ImageProcessor(use_ocr=True, use_multimodal=False)

    def load(self, file_path: str) -> List[Document]:
        """加载图片文件"""
        try:
            # 处理图片，提取文字和描述
            content = self.image_processor.process_image(file_path)

            # 创建Document
            document = Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'file_type': 'image',
                    'file_name': Path(file_path).name,
                    'has_ocr_text': len(content) > 0,
                    'image_dimensions': self._get_image_dimensions(file_path)
                }
            )

            logger.info(f"成功加载图片文件: {file_path}")
            return [document]

        except Exception as e:
            logger.error(f"加载图片文件失败: {file_path}, 错误: {str(e)}")
            raise

    def _get_image_dimensions(self, file_path: str) -> str | None:
        """获取图片尺寸"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                return f"{img.width}x{img.height}"
        except Exception:
            return None

    def supports(self, file_extension: str) -> bool:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        return file_extension.lower() in image_extensions


class PDFImageLoader(PDFLoader):
    """支持图片提取的PDF加载器"""

    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor(use_ocr=True, use_multimodal=False)

    def load(self, file_path: str) -> List[Document]:
        """加载PDF文档，同时提取图片内容"""
        documents = super().load(file_path)

        # 提取PDF中的图片并处理
        image_documents = self._extract_images_from_pdf(file_path)
        documents.extend(image_documents)

        return documents

    def _extract_images_from_pdf(self, file_path: str) -> List[Document]:
        """
        从PDF中提取图片
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            图片内容的Document列表
        """
        try:
            import fitz
            doc = fitz.open(file_path)
            image_docs = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # 临时保存图片
                    temp_dir = Path(file_path).parent / ".temp_images"
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / f"page_{page_num + 1}_img_{img_index + 1}.png"

                    with open(temp_path, 'wb') as f:
                        f.write(image_bytes)

                    # 处理图片内容
                    content = self.image_processor.process_image(str(temp_path))

                    if content:
                        image_doc = Document(
                            page_content=content,
                            metadata={
                                'source': file_path,
                                'file_type': 'pdf_image',
                                'file_name': Path(file_path).name,
                                'page_number': page_num + 1,
                                'image_index': img_index + 1
                            }
                        )
                        image_docs.append(image_doc)

                    # 删除临时文件
                    temp_path.unlink(missing_ok=True)

            if image_docs:
                logger.info(f"从PDF中提取了 {len(image_docs)} 张图片")

            return image_docs

        except ImportError:
            logger.warning("未安装 PyMuPDF，无法提取PDF中的图片")
            return []
        except Exception as e:
            logger.error(f"提取PDF图片失败: {file_path}, 错误: {str(e)}")
            return []


class DocxImageLoader(DocxLoader):
    """支持图片提取的Word加载器"""

    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor(use_ocr=True, use_multimodal=False)

    def load(self, file_path: str) -> List[Document]:
        """加载Word文档，同时提取图片内容"""
        documents = super().load(file_path)

        # 提取Word中的图片并处理
        image_documents = self._extract_images_from_docx(file_path)
        documents.extend(image_documents)

        return documents

    def _extract_images_from_docx(self, file_path: str) -> List[Document]:
        """
        从Word文档中提取图片
        
        Args:
            file_path: Word文件路径
            
        Returns:
            图片内容的Document列表
        """
        try:
            import zipfile

            from docx import Document as DocxDocument

            docx = DocxDocument(file_path)
            image_docs = []
            image_index = 0

            # Word文档是zip格式，可以直接解压提取图片
            with zipfile.ZipFile(file_path, 'r') as zf:
                for name in zf.namelist():
                    if name.startswith('word/media/') and name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        image_index += 1

                        # 临时保存图片
                        temp_dir = Path(file_path).parent / ".temp_images"
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / f"img_{image_index}{Path(name).suffix}"

                        with zf.open(name) as src, open(temp_path, 'wb') as dst:
                            dst.write(src.read())

                        # 处理图片内容
                        content = self.image_processor.process_image(str(temp_path))

                        if content:
                            image_doc = Document(
                                page_content=content,
                                metadata={
                                    'source': file_path,
                                    'file_type': 'docx_image',
                                    'file_name': Path(file_path).name,
                                    'image_index': image_index
                                }
                            )
                            image_docs.append(image_doc)

                        # 删除临时文件
                        temp_path.unlink(missing_ok=True)

            if image_docs:
                logger.info(f"从Word中提取了 {len(image_docs)} 张图片")

            return image_docs

        except ImportError:
            logger.warning("未安装 python-docx，无法提取Word中的图片")
            return []
        except Exception as e:
            logger.error(f"提取Word图片失败: {file_path}, 错误: {str(e)}")
            return []


class ExcelLoader(BaseDocumentLoader):
    """Excel 加载器（按 sheet + 行块切分，保留表头上下文）"""

    def load(self, file_path: str) -> List[Document]:
        import openpyxl

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        file_name = Path(file_path).name
        documents = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))

            if not rows:
                continue

            # 第一行作为表头
            headers = [str(h).strip() if h is not None else "" for h in rows[0]]
            data_rows = rows[1:]

            # 按 10 行一组 + 表头作为一个 chunk
            batch_size = 10
            for batch_start in range(0, len(data_rows), batch_size):
                batch = data_rows[batch_start:batch_start + batch_size]

                lines = [f"【Sheet: {sheet_name} | 表头: {', '.join(headers)}】"]
                for row_idx, row in enumerate(batch, start=batch_start + 2):
                    row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                    lines.append(f"行{row_idx}: {row_text}")

                content = "\n".join(lines)
                if len(content.strip()) < 10:
                    continue

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "sheet_name": sheet_name,
                        "row_start": batch_start + 2,
                        "loader": "excel",
                    }
                ))

        logger.info(f"ExcelLoader: {file_path} → {len(documents)} 个 chunk")
        wb.close()
        return documents

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.xlsx', '.xlsm']


class CSVLoader(BaseDocumentLoader):
    """CSV 加载器（按表头 + 行块切分）"""

    def load(self, file_path: str) -> List[Document]:
        import csv

        file_name = Path(file_path).name
        documents = []

        # 自动检测编码
        encoding = "utf-8"
        for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(file_path, encoding=enc) as f:
                    f.read()
                encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        with open(file_path, encoding=encoding, newline='') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                return []

            data_rows = list(reader)

        batch_size = 10
        for batch_start in range(0, len(data_rows), batch_size):
            batch = data_rows[batch_start:batch_start + batch_size]

            lines = [f"【CSV: {file_name} | 表头: {', '.join(headers)}】"]
            for row_idx, row in enumerate(batch, start=batch_start + 2):
                row_text = " | ".join(row)
                lines.append(f"行{row_idx}: {row_text}")

            content = "\n".join(lines)
            if len(content.strip()) < 10:
                continue

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "row_start": batch_start + 2,
                    "loader": "csv",
                }
            ))

        logger.info(f"CSVLoader: {file_path} → {len(documents)} 个 chunk")
        return documents

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.csv'


class PPTXLoader(BaseDocumentLoader):
    """PPTX 加载器（按幻灯片切分，每页一个 Document）"""

    def load(self, file_path: str) -> List[Document]:
        from pptx import Presentation

        file_name = Path(file_path).name
        prs = Presentation(file_path)
        documents = []

        for slide_idx, slide in enumerate(prs.slides, start=1):
            parts = []

            # 提取文本框内容
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            parts.append(text)

                # 提取表格
                if shape.has_table:
                    table_texts = []
                    for row in shape.table.rows:
                        row_cells = [cell.text.strip() for cell in row.cells]
                        table_texts.append(" | ".join(row_cells))
                    if table_texts:
                        parts.append("\n".join(table_texts))

            content = "\n".join(parts).strip()
            if len(content) < 10:
                continue

            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "slide_number": slide_idx,
                    "loader": "pptx",
                }
            ))

        logger.info(f"PPTXLoader: {file_path} → {len(documents)} 页有效内容")
        return documents

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.pptx', '.ppt']


class DocumentLoaderFactory:
    """文档加载器工厂"""

    _loaders: List[BaseDocumentLoader] = [
        PDFImageLoader(),
        DocxImageLoader(),
        TextLoader(),
        ImageLoader(),
        ExcelLoader(),
        CSVLoader(),
        PPTXLoader(),
    ]

    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """
        根据文件路径获取对应的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档加载器
            
        Raises:
            ValueError: 不支持的文件类型
        """
        file_extension = Path(file_path).suffix.lower()

        for loader in cls._loaders:
            if loader.supports(file_extension):
                return loader

        raise ValueError(f"不支持的文件类型: {file_extension}")

    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """
        加载文档（便捷方法）
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document 列表
        """
        loader = cls.get_loader(file_path)
        return loader.load(file_path)

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取支持的文件扩展名列表"""
        extensions = []
        for loader in cls._loaders:
            if isinstance(loader, (PDFLoader, PDFImageLoader)):
                extensions.append('.pdf')
            elif isinstance(loader, (DocxLoader, DocxImageLoader)):
                extensions.extend(['.docx', '.doc'])
            elif isinstance(loader, TextLoader):
                extensions.extend(['.txt', '.md', '.markdown', '.py', '.java',
                                 '.cpp', '.c', '.h', '.js', '.ts', '.html',
                                 '.css', '.json', '.xml', '.yaml', '.yml'])
            elif isinstance(loader, ImageLoader):
                extensions.extend(['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'])
            elif isinstance(loader, ExcelLoader):
                extensions.extend(['.xlsx', '.xlsm'])
            elif isinstance(loader, CSVLoader):
                extensions.append('.csv')
            elif isinstance(loader, PPTXLoader):
                extensions.extend(['.pptx', '.ppt'])
        return extensions


# 便捷函数
def load_document(file_path: str) -> List[Document]:
    """
    加载单个文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        Document 列表
    """
    return DocumentLoaderFactory.load_document(file_path)


def load_documents_from_directory(directory: str, recursive: bool = True) -> List[Document]:
    """
    从目录加载所有支持的文档
    
    Args:
        directory: 目录路径
        recursive: 是否递归子目录
        
    Returns:
        Document 列表
    """
    from pathlib import Path

    documents = []
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    supported_extensions = DocumentLoaderFactory.get_supported_extensions()

    # 构建搜索模式
    if recursive:
        files = dir_path.rglob("*")
    else:
        files = dir_path.glob("*")

    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_document(str(file_path))
                documents.extend(docs)
                logger.info(f"已加载: {file_path}")
            except Exception as e:
                logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")

    logger.info(f"总共加载了 {len(documents)} 个文档片段")
    return documents
