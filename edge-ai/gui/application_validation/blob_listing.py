from .constants import MAX_BLOBS_SAFETY


def quick_count_blobs(container_client, prefix: str, limit: int = MAX_BLOBS_SAFETY + 1, page_size: int = 5000):
    """Count blobs under prefix up to limit, exiting early at limit."""
    prefix = prefix or ""
    n = 0
    try:
        pager = container_client.list_blobs(name_starts_with=prefix, results_per_page=page_size).by_page()
    except TypeError:
        pager = container_client.list_blobs(name_starts_with=prefix).by_page()
    for page in pager:
        for _ in page:
            n += 1
            if n >= limit:
                return n
    return n
