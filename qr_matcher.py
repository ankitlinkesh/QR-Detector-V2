from text_filter import normalize_text


class QRMatcher:
    """Small helper for exact target matching with --equals."""

    def __init__(self, target_text=None):
        self.target_text = normalize_text(target_text) if target_text else None

    def match(self, decoded_text):
        if self.target_text is None:
            return None

        return normalize_text(decoded_text) == self.target_text
