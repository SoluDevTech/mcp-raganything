#!/usr/bin/env python3
"""
Patch RAGAnything parser.py to support TXT files.
This script patches the installed RAGAnything library to accept .txt, .text, and .md files.

RAGAnything's DoclingParser rejects TXT files even though docling 2.84.0 supports them.
This patch routes TXT files to the existing office document parser which calls docling.
"""

import sys
from pathlib import Path

def patch_raganything():
    """Patch RAGAnything's DoclingParser to support TXT files."""
    
    # Find raganything installation
    try:
        import raganything
        parser_file = Path(raganything.__file__).parent / "parser.py"
    except ImportError:
        print("❌ RAGAnything not found")
        return False
    
    if not parser_file.exists():
        print(f"❌ Parser file not found: {parser_file}")
        return False
    
    print(f"📄 Patching: {parser_file}")
    
    with open(parser_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'TXT files are supported by docling' in content:
        print("✅ Already patched!")
        return True
    
    # Find and patch the format check in DoclingParser.parse_document
    old_code = """        elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Docling only supports PDF files, Office formats ({', '.join(self.OFFICE_FORMATS)}) "
                f"and HTML formats ({', '.join(self.HTML_FORMATS)})"
            )"""
    
    new_code = """        elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        elif ext in {".txt", ".text", ".md"}:
            # TXT files are supported by docling via MarkdownDocumentBackend (PR #3161)
            # Docling 2.84.0+ handles these natively - treat as MD and route to docling
            # Use parse_office_doc which calls DocumentConverter.convert()
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Docling only supports PDF files, Office formats ({', '.join(self.OFFICE_FORMATS)}) "
                f"and HTML formats ({', '.join(self.HTML_FORMATS)})"
            )"""
    
    if old_code not in content:
        print("❌ Patch pattern not found - RAGAnything may have changed")
        print("   Searching for alternative pattern...")
        
        # Try alternative pattern
        alt_old = "elif ext in self.HTML_FORMATS:"
        alt_new = """elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        elif ext in {".txt", ".text", ".md"}:
            # TXT files supported by docling via MarkdownDocumentBackend
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.HTML_FORMATS:"""
        
        if alt_old in content:
            print("   Found alternative pattern, applying patch...")
            content = content.replace(alt_old, alt_new, 1)
        else:
            print("❌ Could not find any pattern to patch")
            return False
    else:
        content = content.replace(old_code, new_code)
    
    # Write patched content
    with open(parser_file, 'w') as f:
        f.write(content)
    
    print("✅ RAGAnything patched successfully!")
    print("   TXT files (.txt, .text, .md) are now supported")
    return True

if __name__ == "__main__":
    success = patch_raganything()
    sys.exit(0 if success else 1)