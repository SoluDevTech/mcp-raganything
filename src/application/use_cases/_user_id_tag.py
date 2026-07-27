"""Per-user RAG isolation helper shared by classical indexing use cases."""

from infrastructure.database.rls_context import current_user_id


def tag_documents_with_user_id(documents: list[tuple]) -> list[tuple]:
    """Tag every document tuple with the authenticated ``user_id`` in metadata.

    Each document is ``(content, file_path, metadata_dict)``. When the RLS
    contextvar ``current_user_id`` is set (authenticated request), a
    ``user_id`` key is merged into the metadata so the query use case can
    filter chunks per user. When the contextvar is ``None`` (legacy/tests,
    no auth wired), documents are returned unchanged.

    Args:
        documents: List of ``(content, file_path, metadata)`` tuples.

    Returns:
        The same list, with ``user_id`` merged into each metadata dict when
        the contextvar is set.
    """
    user_id = current_user_id.get()
    if user_id is None:
        return documents
    return [
        (content, file_path, {**meta, "user_id": user_id})
        for content, file_path, meta in documents
    ]
