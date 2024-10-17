from typing import List, Union
import re
import unicodedata
from nltk.stem import SnowballStemmer
from src.app.conf.Regex import  REGEX

class BaseReplacer:
    r"""
    Base class used by all transformations that require
    replacements with a regular expression. You can use this
    base class to create your own transformations.

    Arguments:
    ----------
        regex: Union[str, List[str]]
            raw string or list of raw strings defining the regular expressions to be used
        repl: Union[str, List[str]]
            string or list of strings to replace the patterns found by the regular expressions.
            If regex is a list of raw strings repl must be a list of string with the same length
    """

    def __init__(
        self,
        regex: Union[str, List[str]],
        repl: Union[str, List[str]],
    ):

        self.regex = regex
        self.repl = repl

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """

        if isinstance(self.regex, list):
            for _regex, _repl in zip(self.regex, self.repl):
                text = re.sub(_regex, _repl, text)
        else:
            text = re.sub(self.regex, self.repl, text)  # type: ignore

        return text
class ToLower:

    """
    Convert text to lower
    """

    def __call__(self, text):
        """
         text: text to transform
        """
        return text.lower()

class RemoveAccents:
    """
    Remove written accents from input string.
    """

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """
        if not isinstance(text, str):
            # assume utf8 for byte strings
            text = text.decode("utf8")

        norm = unicodedata.normalize("NFD", text)
        result = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")

        return unicodedata.normalize("NFC", result)

class RemovePunctuation:
    r"""
    Remove punctuation from input string. It is strongly advised to avoid
    using this transformation before any other transformations that depend
    on punctuation (such as dates, hours, urls, emails).

    Arguments:
    -----------
        punctuation: str, default="!\"#$&'()*+,-_./:;<=>@[\\]^`{|}~?º"
            string with the punctuation marks that will be removed. Notice that the
            percentage mark (%) is not removed by default and that the double
            quotation (") and the backslash (\) must be included in the string by
            preceding them with a backslash (\).
    """

    def __init__(self, punctuation: str = "!\"#$&'()*+,-_./:;<=>@[\\]^`{|}~?º"):

        self.regex = rf"([{re.escape(punctuation)}])"

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """

        return re.sub(self.regex, " ", text)

class RemoveEscapeSequences:
    r"""
    Remove Escape Sequences such as \n \r and \t from input string.

    Arguments:
    ---------
        escape_sequences: List[str], default=None
            list of raw string with the scape sequences to be removed. If None the following
            list is used: [r"\\n", r"\n", r"\\t", r"\t", r"\\a", r"\a", r"\\r",r"\r"].

    Attributes:
    -----------
        regex: str
            regular expression used to remove scape sequences.
    """

    def __init__(self, escape_sequences: List[str] = None):

        escape_sequences = escape_sequences or [
            r"\\n",
            r"\n",
            r"\\t",
            r"\t",
            r"\\a",
            r"\a",
            r"\\r",
            r"\r",
        ]
        self.regex = r"|".join(escape_sequences)

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """

        return re.sub(self.regex, " ", text)

class RemoveHTML:
    """
    Removes HTML tags from input text

    Attributes:
    -----------
        regex: str
            regular expression used to remove HTML tags.
    """

    def __init__(self) -> None:

        self.regex = REGEX["HTML"]

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """

        text = re.sub(self.regex, " ", text)

        return text

class RemoveEmojis:
    """
    Removes emojis from input text

    Attributes:
    -----------
        regex: str
            regular expression used to remove emojis.
    """

    def __init__(self) -> None:

        self.regex = REGEX["EMOJIS"][0]

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """

        text = re.sub(self.regex, " ", text)

        return text

class ReplaceDates(BaseReplacer):
    """
    Replaces dates in a string with a token

    Arguments:
    -----------
        repl: str, default=" %fec "
            replacement string to substitute dates.

    Attributes:
    -----------
        regex: str
            regular expression used to find dates.
        repl: str
            string used to replace dates
    """

    def __init__(self, repl: str = " %fec "):

        self.regex = REGEX["DATES"]
        self.repl = repl

        super().__init__(self.regex, self.repl)

class ReplaceURLs(BaseReplacer):
    """
    Replace urls in a string with a token

    Arguments:
    -----------
        repl: str, default=" %url "
            replacement string to substitute urls.

    Attributes:
    -----------
        regex: str
            regular expression used to find urls.
        repl: str
            string used to replace urls
    """

    def __init__(self, repl: str = " %url "):

        self.regex = REGEX["URLS"]
        self.repl = repl

        super().__init__(self.regex, self.repl)
class RemoveBlankSpaces:
    """
    Removes leading, trailing and duplicated white spaces. This should be
    the last transformation to be applied because some other transformations may
    leave undesired white spaces.
    """

    def __call__(self, text: str) -> str:
        """
        Arguments:
        ----------
            text: unicode string to transform
        """

        text = re.sub(r"^\s+|\s+$", "", text)
        text = re.sub(r" +", " ", text)

        return text
