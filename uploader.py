# uploader.py
import os, datetime as dt
from azure.storage.blob import BlobClient, ContentSettings

# 공통 인증 (연결문자열 권장)
CONN = os.getenv("AZURE_BLOB_CONN_STR", "")

# 컨테이너 이름 (기본값)
CONT_FULL = os.getenv("AZURE_BLOB_CONTAINER_FULL", "intrusion-images")
CONT_CROP = os.getenv("AZURE_BLOB_CONTAINER_CROP", "intrusion-crop-images")

# 컨테이너별 SAS(선택). 둘 다 비우면 CONN으로 동작.
SAS_FULL = os.getenv("AZURE_BLOB_SAS_URL_FULL", "")
SAS_CROP = os.getenv("AZURE_BLOB_SAS_URL_CROP", "")

def _blob_client_for(filename: str, *, container: str, sas_url: str = "") -> BlobClient:
    if CONN:
        return BlobClient.from_connection_string(CONN, container_name=container, blob_name=filename)
    if sas_url:
        base, qs = sas_url.split("?", 1)   # 컨테이너 SAS URL
        return BlobClient.from_blob_url(f"{base.rstrip('/')}/{filename}?{qs}")
    raise RuntimeError("Set AZURE_BLOB_CONN_STR or a SAS URL for the target container")

def upload_file_to(path: str, filename: str, *, container: str, sas_url: str = "") -> str:
    bc = _blob_client_for(filename, container=container, sas_url=sas_url)
    with open(path, "rb") as f:
        bc.upload_blob(f, overwrite=True,
        content_settings=ContentSettings(content_type="image/jpeg"))
    return bc.url

def unique_name(prefix="jetson"):
    ts = dt.datetime.utcnow().isoformat().replace(":", "-")
    return f"{prefix}_{ts}.jpg"