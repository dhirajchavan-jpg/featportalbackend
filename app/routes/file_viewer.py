from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from bson import ObjectId
import os
import mimetypes
from pydantic import BaseModel
from app.config import settings

from app.auth.jwt_bearer import JWTBearer
from app.dependencies import get_current_user, UserPayload
from app.database import Global_file_collection, file_collection
from app.utils.logger import setup_logger

logger = setup_logger()
router = APIRouter(tags=["File Viewer"])

INLINE_EXTENSIONS = {".pdf", ".txt"}


def get_file_serving_metadata(path: str, filename: str = ""):
    path_name = os.path.basename(path or "")
    detected_name = filename or path_name
    extension = os.path.splitext(detected_name)[1].lower()
    if not extension and path_name:
        extension = os.path.splitext(path_name)[1].lower()
        if extension:
            logger.info(
                "[FileViewer] Filename '%s' had no extension. Falling back to path-derived extension '%s' from '%s'.",
                detected_name,
                extension,
                path_name,
            )
            detected_name = detected_name if os.path.splitext(detected_name)[1].lower() else f"{detected_name}{extension}"

    media_type_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
    }

    media_type = media_type_map.get(extension) or mimetypes.guess_type(detected_name)[0] or "application/octet-stream"
    disposition_type = "inline" if extension in INLINE_EXTENSIONS else "attachment"

    logger.info(
        "[FileViewer] Serving metadata resolved. filename=%s, extension=%s, media_type=%s, disposition=%s",
        detected_name,
        extension or "unknown",
        media_type,
        disposition_type,
    )
    return {
        "filename": detected_name,
        "extension": extension,
        "media_type": media_type,
        "content_disposition": f'{disposition_type}; filename="{detected_name}"',
    }


def resolve_physical_path(raw_path: str):
    """Resolve a stored file path against the locations this repo actually uses."""
    if not raw_path:
        return None

    normalized_raw_path = os.path.normpath(raw_path.replace("\\", os.sep).replace("/", os.sep))
    cwd = os.getcwd()
    backend_root = cwd if os.path.basename(cwd).lower() == "backend" else os.path.join(cwd, "Backend")
    workspace_root = os.path.dirname(backend_root) if os.path.basename(backend_root).lower() == "backend" else cwd

    if os.path.isabs(normalized_raw_path) and os.path.exists(normalized_raw_path):
        return normalized_raw_path

    filename = os.path.basename(normalized_raw_path)
    name_candidates = [filename]
    if "..pdf" in filename:
        name_candidates.append(filename.replace("..pdf", ".pdf"))

    final_names = []
    for name in name_candidates:
        final_names.append(name)
        final_names.append(name.replace("_", " "))
        final_names.append(name.replace(" ", "_"))
    final_names = list(dict.fromkeys(final_names))

    relative_candidates = [normalized_raw_path]
    if normalized_raw_path.startswith(f"uploads{os.sep}"):
        relative_candidates.append(normalized_raw_path[len("uploads" + os.sep):])
    if normalized_raw_path.startswith(f"global{os.sep}"):
        relative_candidates.append(os.path.join("uploads", normalized_raw_path))

    base_dirs = [
        cwd,
        backend_root,
        workspace_root,
        os.path.join(cwd, "uploads"),
        os.path.join(cwd, "docs"),
        os.path.join(backend_root, "uploads"),
        os.path.join(backend_root, "docs"),
        os.path.join(workspace_root, "uploads"),
        os.path.join(workspace_root, "docs"),
        os.path.dirname(normalized_raw_path),
    ]
    base_dirs = [os.path.normpath(path) for path in base_dirs if path]
    base_dirs = list(dict.fromkeys(base_dirs))

    for candidate in relative_candidates:
        for base_dir in base_dirs:
            candidate_path = candidate if os.path.isabs(candidate) else os.path.join(base_dir, candidate)
            candidate_path = os.path.normpath(candidate_path)
            if os.path.exists(candidate_path):
                return candidate_path

    for base_dir in base_dirs:
        for name in final_names:
            candidate_path = os.path.normpath(os.path.join(base_dir, name))
            if os.path.exists(candidate_path):
                return candidate_path

    recursive_roots = [
        os.path.join(backend_root, "uploads"),
        os.path.join(workspace_root, "uploads"),
        backend_root,
    ]
    recursive_roots = [os.path.normpath(path) for path in recursive_roots if os.path.isdir(path)]
    recursive_roots = list(dict.fromkeys(recursive_roots))

    for root in recursive_roots:
        for current_root, _, files in os.walk(root):
            lower_files = {file_name.lower(): file_name for file_name in files}
            for name in final_names:
                matched_name = lower_files.get(name.lower())
                if matched_name:
                    return os.path.join(current_root, matched_name)

    return None


class FileNameRequest(BaseModel):
    filename: str


async def get_user_flexible(
    request: Request,
    token: str = Query(None)
):
    if "Authorization" in request.headers:
        token_str = await JWTBearer()(request)
        return get_current_user(token=token_str)

    if token:
        return get_current_user(token=token)

    raise HTTPException(status_code=401, detail="Not authenticated")


async def _find_file_doc_by_filename(original_name: str):
    async def find_in_db(name_to_search: str):
        doc = await Global_file_collection.find_one({"filename": name_to_search})
        if not doc:
            doc = await file_collection.find_one({"filename": name_to_search})
        return doc

    candidates = [original_name]
    if "." in original_name:
        no_ext = original_name.rsplit(".", 1)[0]
        candidates.append(no_ext)
        candidates.append(no_ext.rstrip("."))

    final_candidates = []
    for name in candidates:
        final_candidates.append(name)
        final_candidates.append(name.replace(" ", "_"))
        final_candidates.append(name.replace("_", " "))
    final_candidates = list(dict.fromkeys(final_candidates))

    for candidate in final_candidates:
        file_doc = await find_in_db(candidate)
        if file_doc:
            return file_doc

    return None


@router.post("/files/get-file-path")
async def get_file_path(
    request: Request,
    payload: FileNameRequest,
    current_user: UserPayload = Depends(get_current_user)
):
    file_doc = await _find_file_doc_by_filename(payload.filename)
    if not file_doc:
        logger.warning(">>> File not found in DB after fuzzy search <<<")
        return JSONResponse(status_code=404, content={"message": "File not found"})

    file_id = str(file_doc.get("file_id") or file_doc.get("_id"))
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if "Bearer " in auth_header else ""
    base_url = settings.BACKEND_PUBLIC_URL.rstrip("/")
    secure_url = f"{base_url}/files/view/{file_id}?token={token}"
    file_name = payload.filename or file_doc.get("filename") or ""
    serving_meta = get_file_serving_metadata(file_doc.get("file_path") or file_name, file_name)
    logger.info(
        "[FileViewer] Generated secure file URL. filename=%s, file_id=%s, file_type=%s, url=%s",
        file_name,
        file_id,
        serving_meta["extension"] or "unknown",
        secure_url,
    )
    return {"file_url": secure_url, "file_type": serving_meta["extension"], "filename": file_name}


@router.get("/files/view/{file_id}")
async def view_file(
    file_id: str,
    current_user: UserPayload = Depends(get_user_flexible),
):
    file_doc = None

    try:
        file_doc = await Global_file_collection.find_one({"file_id": file_id})
        if not file_doc:
            file_doc = await Global_file_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            file_doc = await file_collection.find_one({"file_id": file_id})
        if not file_doc:
            file_doc = await file_collection.find_one({"_id": ObjectId(file_id)})
    except Exception:
        pass

    if not file_doc:
        raise HTTPException(status_code=404, detail="File record not found")

    raw_path = file_doc.get("file_path") or file_doc.get("file_url")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Path missing in database record")

    final_path = resolve_physical_path(raw_path)
    if not final_path:
        logger.error(f"ERROR: File '{raw_path}' not found on disk (Checked variations).")
        raise HTTPException(status_code=404, detail="File content not found on server")

    file_name = file_doc.get("filename") or os.path.basename(final_path)
    serving_meta = get_file_serving_metadata(final_path, file_name)
    logger.info(
        "[FileViewer] Opening citation file. requested_file_id=%s, filename=%s, final_path=%s, media_type=%s, content_disposition=%s",
        file_id,
        file_name,
        final_path,
        serving_meta["media_type"],
        serving_meta["content_disposition"],
    )

    def iter_file():
        with open(final_path, "rb") as file_handle:
            while chunk := file_handle.read(8192):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=serving_meta["media_type"],
        headers={"Content-Disposition": serving_meta["content_disposition"]},
    )
