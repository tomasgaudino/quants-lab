def ensure_timestamp_in_seconds(timestamp: float) -> float:
    timestamp_int = int(float(timestamp))
    if timestamp_int >= 1e18:  # Nanoseconds
        return timestamp_int / 1e9
    elif timestamp_int >= 1e15:  # Microseconds
        return timestamp_int / 1e6
    elif timestamp_int >= 1e12:  # Milliseconds
        return timestamp_int / 1e3
    elif timestamp_int >= 1e9:  # Seconds
        return timestamp_int
    else:
        raise ValueError(
            "Timestamp is not in a recognized format. Must be in seconds, milliseconds, microseconds or nanoseconds.")
