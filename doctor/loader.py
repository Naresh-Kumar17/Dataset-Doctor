"""
Robust CSV loader for Dataset Doctor.
Handles encoding detection, 50MB file size limits, keep_default_na=False,
and returns clear, user-friendly errors on invalid or malformed files.
"""
from io import BytesIO, StringIO
from typing import BinaryIO, Tuple
import chardet
import pandas as pd

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class LoaderError(Exception):
    """Base class for CSV loading exceptions."""
    pass


class FileTooLargeError(LoaderError):
    """Raised when an uploaded file exceeds the maximum size limit."""
    pass


class EmptyFileError(LoaderError):
    """Raised when an uploaded file contains 0 bytes or only whitespace."""
    pass


class NoDataRowsError(LoaderError):
    """Raised when a CSV contains headers but zero data rows."""
    pass


class EncodingDetectionError(LoaderError):
    """Raised when the file encoding cannot be decoded."""
    pass


class MalformedCSVError(LoaderError):
    """Raised when the CSV syntax or structure is unparseable."""
    pass


def detect_encoding(raw_bytes: bytes) -> str:
    """
    Detects encoding using UTF-8 test first, falling back to chardet and latin-1.
    """
    # 1. Quick check for valid UTF-8
    try:
        raw_bytes[:65536].decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # 2. Use chardet on a sample
    sample = raw_bytes[:65536]
    detected = chardet.detect(sample)
    encoding = detected.get("encoding")
    confidence = detected.get("confidence", 0.0)

    if encoding and confidence > 0.6:
        try:
            sample.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            pass

    # 3. Fallback to latin-1 (guaranteed single-byte fallback for Western European)
    try:
        sample.decode("latin-1")
        return "latin-1"
    except Exception:
        raise EncodingDetectionError("Could not determine file character encoding. Please provide a UTF-8 or Latin-1 encoded CSV.")


def load_csv(
    file_input: bytes | BinaryIO | str,
    filename: str = "dataset.csv",
    max_size: int = MAX_FILE_SIZE_BYTES
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads and validates a CSV file into a pandas DataFrame.
    
    Rules:
    - Enforces max_size limit (default 50 MB).
    - Uses keep_default_na=False to avoid pandas silently converting 'NA' to NaN.
    - Guesses encoding (UTF-8, detected, or Latin-1).
    - Detects delimiter when non-standard (e.g. semicolon, tab).
    - Catches empty files and header-only files with clear exceptions.
    """
    # Normalize to bytes
    if isinstance(file_input, bytes):
        raw_bytes = file_input
    elif hasattr(file_input, "read"):
        raw_bytes = file_input.read()
    elif isinstance(file_input, str):
        # Could be path or text
        with open(file_input, "rb") as f:
            raw_bytes = f.read()
    else:
        raise LoaderError(f"Unsupported input type: {type(file_input)}")

    # 1. Size check
    size_bytes = len(raw_bytes)
    if size_bytes > max_size:
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise FileTooLargeError(
            f"File '{filename}' ({size_mb:.1f} MB) exceeds the maximum allowed size of {max_mb:.0f} MB."
        )

    # 2. Empty check
    if size_bytes == 0 or len(raw_bytes.strip()) == 0:
        raise EmptyFileError(f"File '{filename}' is empty (0 bytes).")

    # 3. Encoding detection
    encoding = detect_encoding(raw_bytes)

    # 4. Decode full text
    try:
        text_content = raw_bytes.decode(encoding)
    except UnicodeDecodeError as e:
        # If detected encoding failed on full file, try latin-1
        try:
            text_content = raw_bytes.decode("latin-1")
            encoding = "latin-1"
        except Exception:
            raise EncodingDetectionError(f"Failed to decode '{filename}' with {encoding}: {e}")

    # Check for empty lines only
    if not text_content.strip():
        raise EmptyFileError(f"File '{filename}' contains only whitespace or blank lines.")

    # 5. Read with pandas
    # Attempt with default comma first
    try:
        df = pd.read_csv(
            StringIO(text_content),
            keep_default_na=False,
            encoding=encoding,
            on_bad_lines="error"
        )
        delimiter = ","
        # If only 1 column was read, check if lines contain semicolons or tabs
        if len(df.columns) == 1:
            first_line = text_content.strip().splitlines()[0]
            if ";" in first_line and first_line.count(";") > 1:
                df = pd.read_csv(
                    StringIO(text_content),
                    sep=";",
                    keep_default_na=False,
                    on_bad_lines="error"
                )
                delimiter = ";"
            elif "\t" in first_line and first_line.count("\t") > 1:
                df = pd.read_csv(
                    StringIO(text_content),
                    sep="\t",
                    keep_default_na=False,
                    on_bad_lines="error"
                )
                delimiter = "\t"
    except pd.errors.EmptyDataError:
        raise EmptyFileError(f"File '{filename}' contains no parseable data.")
    except Exception as e:
        raise MalformedCSVError(f"Could not parse CSV in '{filename}': {e}")

    # 6. Check for header-only (0 rows)
    if len(df) == 0:
        raise NoDataRowsError(
            f"File '{filename}' contains header columns ({list(df.columns)}) but zero data rows."
        )

    metadata = {
        "filename": filename,
        "size_bytes": size_bytes,
        "encoding": encoding,
        "delimiter": delimiter,
        "rows": len(df),
        "columns": len(df.columns),
    }

    return df, metadata
