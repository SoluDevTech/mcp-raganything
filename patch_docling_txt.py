#!/usr/bin/env python3
"""
Fix docling's _guess_format function to properly detect .txt files as MD format.

Issue: docling's format detection returns None for .txt files instead of InputFormat.MD
Workaround: Patch the _guess_format function to map .txt/.text extensions to MD format.

Based on PR #3161: https://github.com/docling-project/docling/pull/3161
"""

import sys

def patch_docling():
    """Apply monkey-patch to docling's format detection."""
    print("📄 Patching docling format detection for TXT support...")
    
    # Import inside function to ensure packages are available
    from docling.datamodel.document import InputFormat, FormatToExtensions
    
    # Add TXT extensions to MD format
    txt_extensions = ['txt', 'text', 'qmd', 'rmd', 'Rmd']
    
    # Get current MD extensions
    current_md_extensions = FormatToExtensions.get(InputFormat.MD, [])
    
    # Add new extensions if not already present
    for ext in txt_extensions:
        if ext not in current_md_extensions:
            current_md_extensions.append(ext)
    
    FormatToExtensions[InputFormat.MD] = current_md_extensions
    
    print(f"✅ Added TXT extensions to MD format: {txt_extensions}")
    
    # Now patch _guess_format
    import docling.datamodel.document as doc_module
    
    # Get original function
    if hasattr(doc_module, '_guess_format'):
        original_guess = doc_module._guess_format
    else:
        print("⚠️  _guess_format not found, skipping monkey-patch")
        return True
    
    def patched_guess_format(file_path, allowed_formats=None):
        """Version of _guess_format that detects .txt files as MD."""
        from pathlib import Path
        from docling.datamodel.document import InputFormat
        
        path = Path(file_path)
        ext = path.suffix.lower().lstrip('.')
        
        # Map TXT extensions to MD format
        if ext in ['txt', 'text', 'qmd', 'rmd', 'Rmd']:
            result = InputFormat.MD
            if allowed_formats is None or result in allowed_formats:
                return result
        
        # Call original for other formats
        return original_guess(file_path, allowed_formats)
    
    # Apply patch
    doc_module._guess_format = patched_guess_format
    
    print("✅ Monkey-patched _guess_format to handle TXT files")
    return True

if __name__ == "__main__":
    try:
        patch_docling()
        print("✅ Docling TXT support patch applied successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)