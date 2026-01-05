def validate_sql_query(query: str) -> bool:
    query_upper = query.upper().strip()

    dangerous_keywords = ["DROP", "TRUNCATE", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
    if any(keyword in query_upper for keyword in dangerous_keywords):
        return False
    if not query_upper.startswith("SELECT"):
        return False
    return True