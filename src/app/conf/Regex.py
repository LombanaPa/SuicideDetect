REGEX_HTML = r"(?<!\<)(<[^<>]+>)(?!\>)|&([a-z0-9]+|#[0-9]{1,10}|#x[0-9a-f]{1,6});"
REGEX_EMOJIS = [
    "["
    r"\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001f926-\U0001f937"
    r"\U00010000-\U0010ffff"
    r"\u200d"
    r"\u2640-\u2642"
    r"\u2600-\u2B55"
    r"\u23cf"
    r"\u23e9"
    r"\u231a"
    r"\u3030"
    r"\ufe0f"
    "]+"
]
REGEX_URLS = r"(https?\:\s*\/\/\s*|www)\S+"

REGEX = dict(
    HTML = REGEX_HTML,
    EMOJIS = REGEX_EMOJIS,
    URLS = REGEX_URLS
)