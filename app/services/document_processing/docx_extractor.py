import logging
import zipfile
from typing import Any, Dict, List, Tuple

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)


def _iter_block_items(parent):
    """Yield document blocks in source order."""
    if isinstance(parent, DocumentType):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError(f"Unsupported parent type: {type(parent)}")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _attr_by_local_name(element, attr_name: str) -> str | None:
    for key, value in element.attrib.items():
        if _local_name(key) == attr_name:
            return value
    return None


class DocxExtractor:
    """Extracts paginated machine-readable content from DOCX files."""

    def extract_docx(self, file_path: str) -> Dict[str, Any]:
        logger.info("[DocxExtractor] Starting DOCX extraction for file: %s", file_path)
        document = Document(file_path)
        logger.info("[DocxExtractor] DOCX loaded successfully.")

        rendered_break_count = self._count_rendered_page_breaks(file_path)
        logger.info("[DocxExtractor] Rendered page breaks detected: %s", rendered_break_count)

        pages_by_number: Dict[int, Dict[str, Any]] = {}
        current_sections: Dict[int, Dict[str, Any]] = {}
        tables: List[Dict[str, Any]] = []
        title = ""
        block_count = 0
        paragraph_count = 0
        table_count = 0
        current_page_number = 1
        has_meaningful_content = False

        for block in _iter_block_items(document):
            block_count += 1
            if isinstance(block, Paragraph):
                paragraph_count += 1
                style_name = ""
                if block.style is not None and getattr(block.style, "name", None):
                    style_name = block.style.name.strip()

                block_type = self._classify_paragraph(style_name)
                segments, break_count = self._split_paragraph_by_rendered_page_breaks(block)
                logger.info(
                    "[DocxExtractor] Paragraph block %s classified as '%s' with style '%s'. segments=%s, page_breaks=%s",
                    paragraph_count,
                    block_type,
                    style_name or "none",
                    len(segments),
                    break_count,
                )

                for segment_index, segment in enumerate(segments):
                    text = segment.strip()
                    if not text:
                        continue

                    has_meaningful_content = True
                    page_number = current_page_number + segment_index
                    page = self._get_or_create_page(pages_by_number, page_number)
                    self._append_paragraph_to_page(
                        page,
                        current_sections,
                        page_number,
                        block_type,
                        style_name,
                        text,
                    )

                    if not title and block_type in {"title", "heading", "section-header"}:
                        title = text
                        logger.info(
                            "[DocxExtractor] Title initialized from paragraph block %s on page %s: %s",
                            paragraph_count,
                            page_number,
                            title,
                        )

                current_page_number += break_count

            elif isinstance(block, Table):
                table_count += 1
                page_number = current_page_number
                logger.info(
                    "[DocxExtractor] Table block %s detected on page %s. Extracting structured rows.",
                    table_count,
                    page_number,
                )
                table_data = self._extract_table(block, len(tables), page_number)
                tables.append(table_data)

                page = self._get_or_create_page(pages_by_number, page_number)
                page["tables"].append(table_data)

                table_text = self._table_text(table_data)
                if table_text:
                    has_meaningful_content = True
                    page["text_parts"].append(table_text)
                    current_section = current_sections.get(page_number)
                    if current_section is None:
                        current_section = {"type": "table", "elements": [], "text": ""}
                        page["sections"].append(current_section)
                        current_sections[page_number] = current_section

                    current_section["elements"].append(
                        {"type": "table", "table_index": table_data["table_index"], "text": table_text}
                    )
                    current_section["text"] = (
                        f"{current_section['text']}\n{table_text}".strip()
                        if current_section["text"]
                        else table_text
                    )

                current_page_number += self._count_descendant_page_breaks(block._element)

        if not has_meaningful_content:
            logger.info("[DocxExtractor] No meaningful content detected in DOCX.")
            return {
                "pages": [],
                "tables": [],
                "document_structure": {
                    "title": "",
                    "sections": [],
                    "has_toc": False,
                    "section_hierarchy": [],
                },
                "metadata": {
                    "ocr_engine": "none",
                    "extraction_method": "docx_native_paginated",
                    "total_pages": 0,
                    "total_tables": 0,
                    "page_numbering": "physical_docx",
                    "page_source_path": file_path,
                    "logical_source_path": file_path,
                    "source_format": "docx",
                },
            }

        if rendered_break_count == 0:
            raise ValueError(
                "Exact DOCX page numbers are unavailable because the file has no rendered page markers."
            )

        if not title:
            for page_number in sorted(pages_by_number):
                first_text = next((part for part in pages_by_number[page_number]["text_parts"] if part.strip()), "")
                if first_text:
                    title = first_text
                    logger.info("[DocxExtractor] Title fallback selected from page %s.", page_number)
                    break

        pages: List[Dict[str, Any]] = []
        for page_number in sorted(pages_by_number):
            page_data = pages_by_number[page_number]
            text_content = "\n\n".join(page_data["text_parts"]).strip()
            if not text_content and not page_data["tables"] and not page_data["sections"]:
                continue
            pages.append(
                {
                    "page_number": page_number,
                    "layout_elements": [],
                    "text_content": text_content,
                    "ocr_confidence": 0.0,
                    "tables": page_data["tables"],
                    "sections": page_data["sections"],
                }
            )

        document_sections = []
        for page in pages:
            for section in page.get("sections", []):
                if section.get("text"):
                    document_sections.append(
                        {
                            "page": page["page_number"],
                            "type": section.get("type", "paragraph"),
                            "text": section.get("text", ""),
                        }
                    )

        result = {
            "pages": pages,
            "tables": tables,
            "document_structure": {
                "title": title,
                "sections": document_sections,
                "has_toc": False,
                "section_hierarchy": [],
            },
            "metadata": {
                "ocr_engine": "none",
                "extraction_method": "docx_native_paginated",
                "total_pages": max((page["page_number"] for page in pages), default=0),
                "total_tables": len(tables),
                "page_numbering": "physical_docx",
                "page_source_path": file_path,
                "logical_source_path": file_path,
                "source_format": "docx",
            },
        }
        logger.info(
            "[DocxExtractor] DOCX extraction complete. blocks=%s, paragraphs=%s, tables=%s, pages=%s",
            block_count,
            paragraph_count,
            table_count,
            len(result["pages"]),
        )
        return result

    def _get_or_create_page(self, pages_by_number: Dict[int, Dict[str, Any]], page_number: int) -> Dict[str, Any]:
        if page_number not in pages_by_number:
            pages_by_number[page_number] = {
                "text_parts": [],
                "tables": [],
                "sections": [],
            }
        return pages_by_number[page_number]

    def _append_paragraph_to_page(
        self,
        page: Dict[str, Any],
        current_sections: Dict[int, Dict[str, Any]],
        page_number: int,
        block_type: str,
        style_name: str,
        text: str,
    ) -> None:
        page["text_parts"].append(text)

        if block_type in {"title", "heading", "section-header"}:
            current_section = {
                "type": block_type,
                "elements": [{"type": "paragraph", "style": style_name, "text": text}],
                "text": text,
            }
            page["sections"].append(current_section)
            current_sections[page_number] = current_section
            return

        current_section = current_sections.get(page_number)
        if current_section is None:
            current_section = {"type": "paragraph", "elements": [], "text": ""}
            page["sections"].append(current_section)
            current_sections[page_number] = current_section

        current_section["elements"].append(
            {"type": "paragraph", "style": style_name, "text": text}
        )
        current_section["text"] = (
            f"{current_section['text']}\n{text}".strip() if current_section["text"] else text
        )

    def _count_rendered_page_breaks(self, file_path: str) -> int:
        with zipfile.ZipFile(file_path) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        return document_xml.count("lastRenderedPageBreak")

    def _split_paragraph_by_rendered_page_breaks(self, paragraph: Paragraph) -> Tuple[List[str], int]:
        segments: List[str] = []
        current_parts: List[str] = []

        for node in paragraph._element.iter():
            node_name = _local_name(node.tag)
            if node_name == "t" and node.text:
                current_parts.append(node.text)
            elif node_name == "tab":
                current_parts.append("\t")
            elif node_name == "cr":
                current_parts.append("\n")
            elif node_name == "br":
                br_type = _attr_by_local_name(node, "type")
                if br_type == "page":
                    segments.append("".join(current_parts))
                    current_parts = []
                else:
                    current_parts.append("\n")
            elif node_name == "lastRenderedPageBreak":
                segments.append("".join(current_parts))
                current_parts = []

        segments.append("".join(current_parts))
        break_count = max(len(segments) - 1, 0)
        return segments, break_count

    def _count_descendant_page_breaks(self, element) -> int:
        count = 0
        for node in element.iter():
            node_name = _local_name(node.tag)
            if node_name == "lastRenderedPageBreak":
                count += 1
            elif node_name == "br" and _attr_by_local_name(node, "type") == "page":
                count += 1
        return count

    def _classify_paragraph(self, style_name: str) -> str:
        normalized = (style_name or "").lower()
        if normalized == "title":
            return "title"
        if normalized.startswith("heading"):
            return "heading"
        if "subtitle" in normalized or "section" in normalized:
            return "section-header"
        return "paragraph"

    def _table_text(self, table_data: Dict[str, Any]) -> str:
        headers = table_data.get("headers", [])
        rows = table_data.get("data", [])
        table_lines: List[str] = []
        if headers:
            table_lines.append(" | ".join(str(header) for header in headers if header))
        for row in rows:
            ordered_values = [str(row.get(header, "")) for header in headers] if headers else [
                str(value) for value in row.values()
            ]
            row_text = " | ".join(value for value in ordered_values if value)
            if row_text:
                table_lines.append(row_text)
        return "\n".join(table_lines)

    def _extract_table(self, table: Table, table_index: int, page_number: int) -> Dict[str, Any]:
        logger.info("[DocxExtractor] Parsing table index %s.", table_index)
        raw_rows: List[List[str]] = []
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells]
            if any(values):
                raw_rows.append(values)
        logger.info(
            "[DocxExtractor] Table index %s raw extraction complete. Non-empty rows=%s",
            table_index,
            len(raw_rows),
        )

        headers = raw_rows[0] if raw_rows else []
        data_rows = raw_rows[1:] if len(raw_rows) > 1 else []

        if headers:
            records = [
                {header or f"column_{idx + 1}": row[idx] if idx < len(row) else "" for idx, header in enumerate(headers)}
                for row in data_rows
            ]
        else:
            records = []

        table_result = {
            "page": page_number,
            "table_index": table_index,
            "data": records,
            "headers": headers,
            "rows": len(records),
            "columns": len(headers),
            "raw_data": raw_rows,
        }
        logger.info(
            "[DocxExtractor] Table index %s structured. headers=%s, records=%s, columns=%s, page=%s",
            table_index,
            len(headers),
            len(records),
            len(headers),
            page_number,
        )
        return table_result


_docx_extractor = None


def get_docx_extractor() -> DocxExtractor:
    global _docx_extractor
    if _docx_extractor is None:
        _docx_extractor = DocxExtractor()
    return _docx_extractor
