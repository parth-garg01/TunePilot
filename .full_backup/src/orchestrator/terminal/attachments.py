"""Multi-Modal Attachments Engine (PRD Sections 8 & 9, Feature Update)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Attachment:
    path: Path
    name: str
    size_bytes: int
    category: str  # "dataset" | "document" | "code" | "config" | "image" | "log" | "other"
    mime_type: str
    is_image: bool = False
    image_metadata: dict[str, Any] = field(default_factory=dict)
    preview: str = ""
    error: Optional[str] = None

    @property
    def human_size(self) -> str:
        b = self.size_bytes
        if b < 1024:
            return f"{b} B"
        for unit in ["KB", "MB", "GB"]:
            b /= 1024.0
            if b < 1024.0:
                return f"{b:.1f} {unit}"
        return f"{b:.1f} TB"


class AttachmentManager:
    """Discovers, parses, validates, and prepares file and image attachments."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    DATASET_EXTENSIONS = {".jsonl", ".parquet", ".csv", ".tsv"}
    DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".rst"}
    CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini"}
    CODE_EXTENSIONS = {".py", ".sh", ".ps1", ".cu", ".cpp", ".c", ".h"}

    def __init__(self, vision_capable: bool = False) -> None:
        self.vision_capable = vision_capable

    def extract_attachment_paths(self, text: str) -> list[Path]:
        """Extract explicit file paths referenced in prompt text (e.g. ./data/train.jsonl, @path, attach <path>)."""
        paths: list[Path] = []

        # Check @path syntax
        at_matches = re.findall(r"@([a-zA-Z0-9_\-.:/\\]+\.[a-zA-Z0-9]+)", text)
        for m in at_matches:
            paths.append(Path(m))

        # Check attach <path>
        attach_matches = re.findall(r"attach\s+([a-zA-Z0-9_\-.:/\\]+\.[a-zA-Z0-9]+)", text, re.IGNORECASE)
        for m in attach_matches:
            p = Path(m)
            if p not in paths:
                paths.append(p)

        # Check raw paths in text (e.g. ./data/train.jsonl, C:/..., /tmp/...)
        raw_matches = re.findall(r"(?:^|\s)([a-zA-Z0-9_\-.:/\\]+\.[a-zA-Z0-9]{2,5})(?:\s|$)", text)
        for m in raw_matches:
            p = Path(m)
            if (p.exists() or p.is_absolute()) and p not in paths:
                paths.append(p)

        return paths


    def inspect(self, path: Path | str) -> Attachment:
        p = Path(path)
        if not p.exists():
            return Attachment(
                path=p,
                name=p.name,
                size_bytes=0,
                category="other",
                mime_type="unknown",
                error=f"File not found: {p}",
            )

        size = p.stat().st_size
        ext = p.suffix.lower()

        is_img = ext in self.IMAGE_EXTENSIONS
        category = "other"
        if is_img:
            category = "image"
        elif ext in self.DATASET_EXTENSIONS:
            category = "dataset"
        elif ext in self.DOC_EXTENSIONS:
            category = "document"
        elif ext in self.CONFIG_EXTENSIONS:
            category = "config"
        elif ext in self.CODE_EXTENSIONS:
            category = "code"
        elif ext in {".log", ".out", ".err"}:
            category = "log"

        img_meta = {}
        if is_img:
            img_meta = self._inspect_image(p)

        # Generate snippet preview
        preview = ""
        if not is_img and size < 500_000:
            try:
                preview = p.read_text(encoding="utf-8", errors="replace")[:1000]
            except Exception as e:
                preview = f"<preview error: {e}>"

        return Attachment(
            path=p,
            name=p.name,
            size_bytes=size,
            category=category,
            mime_type=f"application/{ext.lstrip('.')}" if not is_img else f"image/{ext.lstrip('.')}",
            is_image=is_img,
            image_metadata=img_meta,
            preview=preview,
        )

    def _inspect_image(self, p: Path) -> dict[str, Any]:
        meta = {
            "format": p.suffix.lower().lstrip("."),
            "vision_supported": self.vision_capable,
        }
        try:
            # Try PIL inspection if pillow installed
            from PIL import Image
            with Image.open(p) as img:
                meta["width"] = img.width
                meta["height"] = img.height
                meta["mode"] = img.mode
        except Exception:
            pass
        return meta

    def format_attachments_card(self, attachments: list[Attachment]) -> str:
        if not attachments:
            return ""

        lines = ["\n[bold cyan]Attachments[/bold cyan]"]
        for a in attachments:
            if a.error:
                lines.append(f"  [red]✗[/red] {a.name:<20} [red]{a.error}[/red]")
            else:
                extra = ""
                if a.is_image:
                    if a.image_metadata.get("vision_supported"):
                        extra = " [green](Vision Enabled)[/green]"
                    else:
                        extra = " [yellow](Image Metadata Mode)[/yellow]"
                lines.append(f"  [green]✓[/green] {a.name:<24} [dim]{a.human_size:>9}[/dim]{extra}")

        lines.append("[dim]Ready to analyze.[/dim]\n")
        return "\n".join(lines)
