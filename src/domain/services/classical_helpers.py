import os


def validate_path(output_dir: str, file_name: str) -> str:
    file_path = os.path.join(output_dir, file_name)
    real_output = os.path.realpath(output_dir)
    real_file = os.path.realpath(file_path)
    if not real_file.startswith(real_output + os.sep) and real_file != real_output:
        raise ValueError(f"file_name escapes output directory: {file_name}")
    return file_path


def build_documents_from_extraction(
    result, file_name: str
) -> list[tuple[str, str, dict[str, int]]]:
    documents = []
    if result.chunks:
        for i, chunk in enumerate(result.chunks):
            documents.append((chunk.content, file_name, {"chunk_index": i}))
    elif result.content and result.content.strip():
        documents.append((result.content, file_name, {}))
    return documents
