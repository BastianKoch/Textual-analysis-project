"""Text processing utilities for earnings call transcripts."""

import os
import re
from pathlib import Path
from typing import Optional

# Automatically set working directory to project root when this module is imported
# This ensures relative paths work correctly regardless of where notebooks/scripts are run from
_current_dir = Path.cwd()
if _current_dir.name == "notebooks" or _current_dir.name.startswith("."):
    # If we're in notebooks folder or hidden folder, go to parent
    os.chdir(_current_dir.parent)
elif not (_current_dir / "src" / "text_processing.py").exists():
    # If src folder doesn't exist in current dir, search up the tree
    for parent in _current_dir.parents:
        if (parent / "src" / "text_processing.py").exists():
            os.chdir(parent)
            break


def load_transcript(filepath: str | Path) -> str:
    """
    Load a single earnings call transcript.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the transcript file (relative or absolute)
        
    Returns
    -------
    str
        Raw transcript text
    """
    filepath = Path(filepath)
    
    # If path doesn't exist, try to find it relative to project root
    if not filepath.exists():
        # Try from current directory first, then from parent directories
        for potential_root in [Path.cwd()] + list(Path.cwd().parents):
            potential_path = potential_root / filepath
            if potential_path.exists():
                filepath = potential_path
                break
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_presentation(text: str) -> str:
    """
    Extract only the presentation/Q&A content from a transcript.
    
    Removes header metadata and participant lists, keeping only
    the actual spoken content.
    
    Parameters
    ----------
    text : str
        Full transcript text
        
    Returns
    -------
    str
        Presentation and Q&A content only
    """
    # Find where the presentation section starts
    match = re.search(r'={10,}\s*Presentation\s*[-=]{10,}', text, re.IGNORECASE)
    if match:
        return text[match.end():]
    return text


def clean_text(text: str, 
               lowercase: bool = True,
               remove_speaker_tags: bool = True,
               remove_extra_whitespace: bool = True) -> str:
    """
    Clean and normalize transcript text.
    
    Parameters
    ----------
    text : str
        Raw transcript text
    lowercase : bool, default=True
        Convert text to lowercase
    remove_speaker_tags : bool, default=True
        Remove speaker names and tags (e.g., "Jeff Lorberbaum, CEO [2]")
    remove_extra_whitespace : bool, default=True
        Collapse multiple spaces/newlines
        
    Returns
    -------
    str
        Cleaned text
    """
    cleaned = text
    
    # Remove speaker tags (lines with [N] at the end)
    if remove_speaker_tags:
        cleaned = re.sub(r'^.*?\[\d+\]\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove dashes used as separators
    cleaned = re.sub(r'^-{10,}$', '', cleaned, flags=re.MULTILINE)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 newlines
        cleaned = re.sub(r' {2,}', ' ', cleaned)       # Single spaces
        cleaned = cleaned.strip()
    
    # Convert to lowercase
    if lowercase:
        cleaned = cleaned.lower()
    
    return cleaned


def process_transcript(input_path: str | Path,
                      output_path: Optional[str | Path] = None,
                      **clean_kwargs) -> str:
    """
    Load, process, and optionally save a single transcript.
    
    Parameters
    ----------
    input_path : str or Path
        Path to raw transcript
    output_path : str or Path, optional
        If provided, save cleaned transcript to this path
    **clean_kwargs
        Additional arguments passed to clean_text()
        
    Returns
    -------
    str
        Cleaned transcript text
    """
    # Load
    text = load_transcript(input_path)
    
    # Extract presentation content
    text = extract_presentation(text)
    
    # Clean
    text = clean_text(text, **clean_kwargs)
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    return text


def process_all_transcripts(input_dir: str | Path,
                           output_dir: str | Path,
                           pattern: str = "**/*.txt",
                           **clean_kwargs) -> dict[str, int]:
    """
    Process all transcripts from raw to interim folder.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw transcripts
    output_dir : str or Path
        Directory to save cleaned transcripts (structure preserved)
    pattern : str, default="**/*.txt"
        Glob pattern to match transcript files
    **clean_kwargs
        Additional arguments passed to clean_text()
        
    Returns
    -------
    dict
        Statistics: {'processed': count, 'failed': count}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    stats = {'processed': 0, 'failed': 0}
    
    for input_file in input_dir.glob(pattern):
        if not input_file.is_file():
            continue
            
        # Preserve directory structure
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        
        try:
            process_transcript(input_file, output_file, **clean_kwargs)
            stats['processed'] += 1
        except Exception as e:
            print(f"Failed to process {input_file}: {e}")
            stats['failed'] += 1
    
    return stats
