import os
import io
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Load .env from: ai app/other-files/.env
dotenv_path = Path(__file__).resolve().parents[2] / "other-files" / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Local storage configuration
LOCAL_STORAGE_ROOT = os.getenv("LOCAL_STORAGE_ROOT", "local_storage")
LOCAL_STORAGE_BUCKET = os.getenv("LOCAL_STORAGE_BUCKET", "documents")

# Create the full storage path
STORAGE_PATH = Path(LOCAL_STORAGE_ROOT) / LOCAL_STORAGE_BUCKET

print(f"📦 Local Storage Configuration:")
print(f"   - Root Path: {LOCAL_STORAGE_ROOT}")
print(f"   - Bucket: {LOCAL_STORAGE_BUCKET}")
print(f"   - Full Path: {STORAGE_PATH}")

# Create storage directory if it doesn't exist
try:
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    print("✅ Local storage directory created/verified")
except Exception as e:
    print(f"❌ Failed to create storage directory: {e}")
    raise e

def _get_object_path(object_name: str) -> Path:
    """Get the full local path for an object"""
    # Ensure object_name doesn't try to escape the storage directory
    object_name = object_name.lstrip('/')
    return STORAGE_PATH / object_name

def _calculate_etag(file_path: Path) -> str:
    """Calculate MD5 hash as etag (similar to MinIO/S3)"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return "unknown"

def upload_file_to_storage(local_path: str, object_name: Optional[str] = None, 
                          folder: Optional[str] = None, overwrite: bool = True) -> Optional[str]:
    """
    Uploads a file to local storage and returns its object name.
    
    Args:
        local_path (str): Path to the local file
        object_name (str, optional): Name for the object in storage. Defaults to filename.
        folder (str, optional): Folder/prefix in the bucket
        overwrite (bool): Whether to overwrite existing files
    
    Returns:
        str: Object name if successful, None if failed
    """
    if not os.path.exists(local_path):
        print(f"❌ Local file not found: {local_path}")
        return None
    
    if not object_name:
        object_name = os.path.basename(local_path)
    
    # Add folder prefix if specified
    if folder:
        folder = folder.strip('/')  # Remove leading/trailing slashes
        object_name = f"{folder}/{object_name}"
    
    try:
        dest_path = _get_object_path(object_name)
        
        # Check if object already exists
        if not overwrite and dest_path.exists():
            print(f"⚠️ Object already exists and overwrite=False: {object_name}")
            return object_name
        
        # Create directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file size for progress tracking
        file_size = os.path.getsize(local_path)
        
        # Copy file
        shutil.copy2(local_path, dest_path)
        
        print(f"✅ Uploaded to local storage: {object_name} ({file_size} bytes)")
        return object_name
        
    except Exception as e:
        print(f"❌ Local storage upload failed: {e}")
        return None

def upload_data_to_storage(data: bytes, object_name: str, 
                          folder: Optional[str] = None, content_type: str = "application/octet-stream") -> Optional[str]:
    """
    Uploads binary data directly to local storage.
    
    Args:
        data (bytes): Binary data to upload
        object_name (str): Name for the object in storage
        folder (str, optional): Folder/prefix in the bucket
        content_type (str): MIME type of the data (stored as metadata)
    
    Returns:
        str: Object name if successful, None if failed
    """
    if folder:
        folder = folder.strip('/')
        object_name = f"{folder}/{object_name}"
    
    try:
        dest_path = _get_object_path(object_name)
        
        # Create directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write data to file
        with open(dest_path, 'wb') as f:
            f.write(data)
        
        # Store metadata (content type) in a separate file
        metadata_path = dest_path.with_suffix(dest_path.suffix + '.meta')
        metadata = {
            'content_type': content_type,
            'size': len(data),
            'created': datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"✅ Uploaded data to local storage: {object_name} ({len(data)} bytes)")
        return object_name
        
    except Exception as e:
        print(f"❌ Local storage data upload failed: {e}")
        return None

def download_file_from_storage(object_name: str, local_path: str) -> bool:
    """
    Downloads a file from local storage and saves it to another location.
    
    Args:
        object_name (str): Name of the object in storage
        local_path (str): Local path where file will be saved
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        source_path = _get_object_path(object_name)
        
        if not source_path.exists():
            print(f"❌ Object not found in storage: {object_name}")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, local_path)
        
        file_size = os.path.getsize(local_path)
        print(f"✅ Downloaded from local storage: {object_name} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print(f"❌ Local storage download failed: {e}")
        return False

def get_file_data_from_storage(object_name: str) -> Optional[bytes]:
    """
    Gets file data directly from local storage as bytes.
    
    Args:
        object_name (str): Name of the object in storage
    
    Returns:
        bytes: File data if successful, None if failed
    """
    try:
        source_path = _get_object_path(object_name)
        
        if not source_path.exists():
            print(f"❌ Object not found in storage: {object_name}")
            return None
        
        with open(source_path, 'rb') as f:
            data = f.read()
        
        print(f"✅ Retrieved data from local storage: {object_name} ({len(data)} bytes)")
        return data
        
    except Exception as e:
        print(f"❌ Local storage data retrieval failed: {e}")
        return None

def list_files_in_storage(prefix: Optional[str] = None, max_keys: int = 1000) -> List[Dict]:
    """
    Lists files in the local storage.
    
    Args:
        prefix (str, optional): Filter objects by prefix
        max_keys (int): Maximum number of objects to return
    
    Returns:
        List[Dict]: List of file information dictionaries
    """
    try:
        files = []
        count = 0
        
        # Walk through storage directory
        for root, dirs, filenames in os.walk(STORAGE_PATH):
            if count >= max_keys:
                break
                
            for filename in filenames:
                if count >= max_keys:
                    break
                
                # Skip metadata files
                if filename.endswith('.meta'):
                    continue
                
                file_path = Path(root) / filename
                
                # Calculate relative path from storage root
                relative_path = file_path.relative_to(STORAGE_PATH)
                object_name = str(relative_path).replace('\\', '/')  # Normalize path separators
                
                # Apply prefix filter
                if prefix and not object_name.startswith(prefix):
                    continue
                
                # Get file stats
                stat = file_path.stat()
                
                files.append({
                    'name': object_name,
                    'size': stat.st_size,
                    'last_modified': datetime.fromtimestamp(stat.st_mtime),
                    'etag': _calculate_etag(file_path)
                })
                count += 1
        
        print(f"📄 Listed {len(files)} files from local storage")
        return files
        
    except Exception as e:
        print(f"❌ Local storage listing failed: {e}")
        return []

def file_exists_in_storage(object_name: str) -> bool:
    """
    Checks if a file exists in local storage.
    
    Args:
        object_name (str): Name of the object to check
    
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        return _get_object_path(object_name).exists()
    except Exception as e:
        print(f"❌ Error checking file existence: {e}")
        return False

def delete_file_from_storage(object_name: str) -> bool:
    """
    Deletes a file from local storage.
    
    Args:
        object_name (str): Name of the object to delete
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        file_path = _get_object_path(object_name)
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        
        # Remove main file
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ Deleted from local storage: {object_name}")
        
        # Remove metadata file if it exists
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove empty directories
        try:
            file_path.parent.rmdir()
        except OSError:
            pass  # Directory not empty, that's fine
        
        return True
        
    except Exception as e:
        print(f"❌ Local storage deletion failed: {e}")
        return False

def get_storage_url(object_name: str, expires: int = 3600) -> Optional[str]:
    """
    Gets a file:// URL for the object (local filesystem access).
    
    Args:
        object_name (str): Name of the object
        expires (int): Not used for local storage, kept for compatibility
    
    Returns:
        str: File URL if successful, None if failed
    """
    try:
        file_path = _get_object_path(object_name)
        
        if not file_path.exists():
            print(f"❌ Object not found: {object_name}")
            return None
        
        # Create file:// URL
        url = file_path.as_uri()
        print(f"🔗 Generated local URL for: {object_name}")
        return url
        
    except Exception as e:
        print(f"❌ URL generation failed: {e}")
        return None

def get_storage_public_url(object_name: str) -> str:
    """
    Gets a local file path for the object.
    
    Args:
        object_name (str): Name of the object
    
    Returns:
        str: Local file path
    """
    return str(_get_object_path(object_name))

def get_bucket_stats() -> Dict:
    """
    Gets statistics about the local storage bucket.
    
    Returns:
        Dict: Bucket statistics
    """
    try:
        total_files = 0
        total_size = 0
        file_types = {}
        
        # Walk through storage directory
        for root, dirs, filenames in os.walk(STORAGE_PATH):
            for filename in filenames:
                # Skip metadata files
                if filename.endswith('.meta'):
                    continue
                
                file_path = Path(root) / filename
                stat = file_path.stat()
                
                total_files += 1
                total_size += stat.st_size
                
                # Extract file extension
                ext = file_path.suffix.lower()
                if not ext:
                    ext = 'no_extension'
                
                file_types[ext] = file_types.get(ext, 0) + 1
        
        stats = {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_types': file_types,
            'bucket_name': LOCAL_STORAGE_BUCKET,
            'storage_path': str(STORAGE_PATH)
        }
        
        return stats
        
    except Exception as e:
        print(f"❌ Error getting bucket stats: {e}")
        return {}

def test_storage_connection() -> bool:
    """
    Tests the local storage operations.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("🧪 Testing local storage connection...")
    
    try:
        # Test 1: Check if storage directory is accessible
        if not STORAGE_PATH.exists():
            print(f"❌ Storage path does not exist: {STORAGE_PATH}")
            return False
        print(f"✅ Storage path accessible: {STORAGE_PATH}")
        
        # Test 2: Test write permissions
        test_dir = STORAGE_PATH / "test"
        test_dir.mkdir(exist_ok=True)
        print("✅ Can create directories")
        
        # Test 3: Upload a test file
        test_data = b"Local storage connection test"
        test_object = "test/connection_test.txt"
        
        upload_result = upload_data_to_storage(test_data, test_object)
        if upload_result:
            print("✅ Test upload successful")
            
            # Test 4: Download the test file
            downloaded_data = get_file_data_from_storage(test_object)
            if downloaded_data == test_data:
                print("✅ Test download successful")
                
                # Test 5: Check file exists
                if file_exists_in_storage(test_object):
                    print("✅ File existence check successful")
                    
                    # Test 6: Delete the test file
                    if delete_file_from_storage(test_object):
                        print("✅ Test deletion successful")
                        
                        # Clean up test directory
                        try:
                            test_dir.rmdir()
                        except OSError:
                            pass
                        
                        print("🎉 All local storage tests passed!")
                        return True
        
        print("❌ Some local storage tests failed")
        return False
        
    except Exception as e:
        print(f"❌ Local storage test failed: {e}")
        return False

# Backward compatibility aliases (maintain same function names as MinIO version)
upload_file_to_minio = upload_file_to_storage
upload_data_to_minio = upload_data_to_storage
download_file_from_minio = download_file_from_storage
get_file_data_from_minio = get_file_data_from_storage
list_files_in_minio = list_files_in_storage
file_exists_in_minio = file_exists_in_storage
delete_file_from_minio = delete_file_from_storage
get_minio_url = get_storage_url
get_minio_public_url = get_storage_public_url
test_minio_connection = test_storage_connection

# Test connection when module is imported
if __name__ == "__main__":
    print("🚀 Testing local storage module...")
    test_storage_connection()
    print("📊 Bucket stats:", get_bucket_stats())
else:
    print("📦 Local storage module loaded")
    # Quick connection test
    try:
        if STORAGE_PATH.exists() and STORAGE_PATH.is_dir():
            print(f"✅ Local storage connection verified, path '{STORAGE_PATH}' accessible")
    except Exception as e:
        print(f"⚠️ Local storage connection issue: {e}")