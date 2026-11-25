"""On-demand Japanese romaji and Chinese pinyin for translated text."""

from __future__ import annotations

import logging
from threading import Lock

logger = logging.getLogger(__name__)
_japanese_lock = Lock()

_KANA = {
    **dict(zip("あいうえお", ("a", "i", "u", "e", "o"))),
    **dict(zip("かきくけこ", ("ka", "ki", "ku", "ke", "ko"))),
    **dict(zip("さしすせそ", ("sa", "shi", "su", "se", "so"))),
    **dict(zip("たちつてと", ("ta", "chi", "tsu", "te", "to"))),
    **dict(zip("なにぬねの", ("na", "ni", "nu", "ne", "no"))),
    **dict(zip("はひふへほ", ("ha", "hi", "fu", "he", "ho"))),
    **dict(zip("まみむめも", ("ma", "mi", "mu", "me", "mo"))),
    **dict(zip("やゆよ", ("ya", "yu", "yo"))),
    **dict(zip("らりるれろ", ("ra", "ri", "ru", "re", "ro"))),
    **dict(zip("わを", ("wa", "o"))),
    **dict(zip("がぎぐげご", ("ga", "gi", "gu", "ge", "go"))),
    **dict(zip("ざじずぜぞ", ("za", "ji", "zu", "ze", "zo"))),
    **dict(zip("だぢづでど", ("da", "ji", "zu", "de", "do"))),
    **dict(zip("ばびぶべぼ", ("ba", "bi", "bu", "be", "bo"))),
    **dict(zip("ぱぴぷぺぽ", ("pa", "pi", "pu", "pe", "po"))),
    **dict(zip("ぁぃぅぇぉ", ("a", "i", "u", "e", "o"))),
    "ゔ": "vu",
    "ゎ": "wa",
    "ゐ": "i",
    "ゑ": "e",
    "ん": "n",
}
_DIGRAPHS = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
}
_MACRON = {"a": "ā", "i": "ī", "u": "ū", "e": "ē", "o": "ō"}


class PronunciationError(RuntimeError):
    """A safe, user-facing error from the optional pronunciation feature."""


def _kana_to_romaji(kana: str) -> str:
    import jaconv

    text = jaconv.kata2hira(kana)
    output: list[str] = []
    small_tsu = False
    index = 0

    while index < len(text):
        character = text[index]
        if character == "っ":
            small_tsu = True
            index += 1
            continue
        if character == "ー":
            for position in range(len(output) - 1, -1, -1):
                syllable = output[position]
                if not syllable or not syllable[-1].isalpha():
                    break
                vowel_position = next(
                    (offset for offset in range(len(syllable) - 1, -1, -1) if syllable[offset] in _MACRON),
                    None,
                )
                if vowel_position is not None:
                    vowel = syllable[vowel_position]
                    output[position] = syllable[:vowel_position] + _MACRON[vowel] + syllable[vowel_position + 1:]
                    break
            index += 1
            continue

        pair = text[index:index + 2]
        if pair in _DIGRAPHS:
            syllable = _DIGRAPHS[pair]
            index += 2
        else:
            syllable = _KANA.get(character, character)
            index += 1

        if character == "ん" and index < len(text):
            next_character = text[index]
            next_syllable = _KANA.get(next_character, "")
            if next_syllable.startswith(("a", "i", "u", "e", "o", "y")):
                syllable = "n'"

        if small_tsu:
            if syllable.startswith("ch"):
                output.append("t")
            elif syllable and syllable[0].isalpha() and syllable[0] not in "aiueoy":
                output.append(syllable[0])
            small_tsu = False

        if character == "、":
            output.append(",")
        elif character == "。":
            output.append(".")
        elif character in {"「", "『"}:
            output.append('"')
        elif character in {"」", "』"}:
            output.append('"')
        else:
            output.append(syllable)

    return "".join(output).replace("\u3000", " ").strip()


def transliterate(text: str, target_language: str) -> str:
    """Return a local reading aid; callers decide when to request it."""
    if target_language in {"zhCN", "zhTW"}:
        try:
            from pypinyin import Style, lazy_pinyin

            return " ".join(lazy_pinyin(text, style=Style.TONE, v_to_u=True))
        except ImportError as error:
            raise PronunciationError(
                "Chinese pronunciation support is missing. Run uv sync and restart FastAPI."
            ) from error
        except Exception as error:
            logger.exception("Could not generate Chinese pinyin")
            raise PronunciationError(
                "Chinese pronunciation could not be generated. Try again."
            ) from error

    if target_language == "ja":
        try:
            import pyopenjtalk

            with _japanese_lock:
                kana = pyopenjtalk.g2p(text, kana=True)
            return _kana_to_romaji(kana)
        except ImportError as error:
            raise PronunciationError(
                "Japanese pronunciation support is missing. Run uv sync and restart FastAPI."
            ) from error
        except Exception as error:
            logger.exception("Could not generate Japanese romaji")
            raise PronunciationError(
                "Japanese pronunciation could not be generated. Try again."
            ) from error

    raise PronunciationError("Pronunciation is available for Japanese and Chinese translations.")
