from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import AppConfig


@dataclass(frozen=True)
class UploadedImage:
    secure_url: str
    public_id: str | None


def upload_image_to_cloudinary(config: AppConfig, image_path: Path) -> UploadedImage:
    try:
        import cloudinary
        import cloudinary.uploader
    except ImportError as exc:
        raise RuntimeError("Cloudinary package is not installed.") from exc

    cloudinary.config(
        cloud_name=config.cloudinary_cloud_name,
        api_key=config.cloudinary_api_key,
        api_secret=config.cloudinary_api_secret,
        secure=True,
    )
    try:
        payload = cloudinary.uploader.upload(
            str(image_path),
            folder=config.cloudinary_folder,
            resource_type="image",
        )
    except Exception as exc:
        raise RuntimeError(f"Cloudinary upload failed: {exc}") from exc

    secure_url = payload.get("secure_url")
    if not isinstance(secure_url, str) or not secure_url.strip():
        raise RuntimeError("Cloudinary upload response did not include secure_url.")
    public_id = payload.get("public_id")
    return UploadedImage(
        secure_url=secure_url,
        public_id=public_id if isinstance(public_id, str) else None,
    )
