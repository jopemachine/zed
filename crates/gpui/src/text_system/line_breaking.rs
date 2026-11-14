use crate::{OverflowWrap, WordBreak};
use icu_properties::{
    CodePointMapData,
    props::{GeneralCategory, GeneralCategoryGroup, LineBreak},
};
use icu_segmenter::LineSegmenter;

/// Where wrapping may break the lines of a text: its CSS `word-break` and `overflow-wrap`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LineBreaking {
    /// Where a wrapped line may break between two characters.
    pub word_break: WordBreak,
    /// Whether a word wider than its line breaks inside.
    pub overflow_wrap: OverflowWrap,
}

impl LineBreaking {
    /// The byte offsets in `text` at which a wrapped line may start, in increasing order, when
    /// the word break follows the Unicode line breaking algorithm. `None` for
    /// [`WordBreak::WordCharacters`], whose opportunities the wrapping code decides from each
    /// character and the one before it.
    ///
    /// The opportunities are those Chromium takes, which tailors UAX #14 in three ways. Between
    /// two printable ASCII characters it breaks only after `-` or `?` and before an opening
    /// bracket, so `a/b` and `(Asia/Seoul)` stay whole. `keep-all` keeps two letters or digits of
    /// any script together and leaves the other opportunities of `normal`, so `#태그` still
    /// breaks after `#`. `break-all` adds opportunities by the line break classes of the two
    /// characters, to those of `normal`.
    pub(crate) fn line_start_offsets(&self, text: &str) -> Option<Vec<usize>> {
        if self.word_break == WordBreak::WordCharacters {
            return None;
        }
        let mut unicode_offsets = LineSegmenter::new_17_for_non_complex_scripts(Default::default())
            .segment_str(text)
            .peekable();
        let line_break_classes = CodePointMapData::<LineBreak>::new();

        let mut offsets = Vec::new();
        let mut before_previous = None;
        let mut previous = None;
        let mut previous_class = LineBreak::Unknown;
        for (offset, character) in text.char_indices() {
            while unicode_offsets
                .next_if(|unicode| *unicode < offset)
                .is_some()
            {}
            let unicode_allows = unicode_offsets.next_if_eq(&offset).is_some();
            let class = if character == '+' {
                LineBreak::Alphabetic
            } else {
                line_break_classes.get(character)
            };
            if let Some(previous) = previous {
                let pair = CharacterPair {
                    before_previous,
                    previous,
                    previous_class,
                    character,
                    class,
                };
                if self.may_start_line_between(&pair, unicode_allows) {
                    offsets.push(offset);
                }
            }
            if class != LineBreak::CombiningMark {
                previous_class = class;
            }
            before_previous = previous;
            previous = Some(character);
        }
        Some(offsets)
    }

    /// Whether `character` at the end of a wrapped line stays on it even past the wrap width, as
    /// a browser lets spaces hang at the end of a line that wraps after them.
    pub(crate) fn hangs_past_line_end(character: char) -> bool {
        Self::is_breakable_space(character) || Self::is_space_separator(character)
    }

    fn may_start_line_between(&self, pair: &CharacterPair, unicode_allows: bool) -> bool {
        if Self::is_breakable_space(pair.character) {
            return false;
        }
        if Self::is_breakable_space(pair.previous) || Self::ascii_pair_breaks(pair) {
            return true;
        }
        if self.word_break == WordBreak::BreakAll
            && Self::break_all_adds_opportunity(pair.previous_class, pair.class)
            && !Self::is_space_separator(pair.character)
        {
            return true;
        }
        if self.word_break == WordBreak::KeepAll && Self::keeps_letters_together(pair) {
            return false;
        }
        let follows_unicode = |character: char| !character.is_ascii() && character != '\u{A0}';
        (follows_unicode(pair.previous) || follows_unicode(pair.character)) && unicode_allows
    }

    fn ascii_pair_breaks(pair: &CharacterPair) -> bool {
        if !Self::is_printable_ascii(pair.previous) || !Self::is_printable_ascii(pair.character) {
            return false;
        }
        match pair.previous {
            '-' if pair.character.is_ascii_digit() => pair
                .before_previous
                .is_some_and(|before| before.is_ascii_alphanumeric()),
            '-' => !matches!(
                pair.character,
                '!' | '$' | ')' | ',' | '.' | '/' | ':' | ';' | '?' | ']' | '}'
            ),
            '?' => !matches!(
                pair.character,
                '!' | '"' | '\'' | ')' | ',' | '.' | '/' | ':' | ';' | '?' | ']' | '}'
            ),
            previous => {
                matches!(pair.character, '(' | '<' | '[' | '{')
                    && !previous.is_ascii_alphanumeric()
                    && !matches!(
                        previous,
                        '$' | '\'' | '/' | '@' | '^' | '_' | '`' | '(' | '<' | '[' | '{'
                    )
            }
        }
    }

    fn break_all_adds_opportunity(previous_class: LineBreak, class: LineBreak) -> bool {
        use LineBreak as Lb;
        let after_letters = [
            Lb::AI,
            Lb::AL,
            Lb::BA,
            Lb::HL,
            Lb::HY,
            Lb::NU,
            Lb::OP,
            Lb::PR,
            Lb::SA,
        ];
        let classes_after: &[LineBreak] = match previous_class {
            Lb::AI | Lb::AL | Lb::HL | Lb::NU | Lb::SA => &after_letters,
            Lb::BA | Lb::CL => &[Lb::AL, Lb::BA, Lb::HY, Lb::NU, Lb::PR],
            Lb::CP => &[
                Lb::AI,
                Lb::AL,
                Lb::BA,
                Lb::HL,
                Lb::HY,
                Lb::NU,
                Lb::PR,
                Lb::SA,
            ],
            Lb::EX => &[Lb::AL, Lb::BA, Lb::HY, Lb::NU, Lb::PO, Lb::PR],
            Lb::HY => &[Lb::NU],
            Lb::IS => &[Lb::AI, Lb::AL, Lb::BA, Lb::HL, Lb::HY, Lb::NU],
            Lb::PO => &[
                Lb::AI,
                Lb::AL,
                Lb::BA,
                Lb::HL,
                Lb::HY,
                Lb::NU,
                Lb::PO,
                Lb::PR,
            ],
            Lb::PR => &[Lb::PO],
            Lb::SY => &[Lb::AL, Lb::BA, Lb::HL, Lb::HY, Lb::NU, Lb::OP, Lb::PR],
            _ => &[],
        };
        classes_after.contains(&class)
    }

    fn keeps_letters_together(pair: &CharacterPair) -> bool {
        let general_categories = CodePointMapData::<GeneralCategory>::new();
        let is_letter_or_digit = |character: char| {
            let category = general_categories.get(character);
            GeneralCategoryGroup::Letter.contains(category)
                || GeneralCategoryGroup::Number.contains(category)
        };
        let base = match pair.before_previous {
            Some(before)
                if GeneralCategoryGroup::Mark.contains(general_categories.get(pair.previous)) =>
            {
                before
            }
            _ => pair.previous,
        };
        is_letter_or_digit(base) && is_letter_or_digit(pair.character)
    }

    fn is_space_separator(character: char) -> bool {
        CodePointMapData::<GeneralCategory>::new().get(character) == GeneralCategory::SpaceSeparator
    }

    fn is_breakable_space(character: char) -> bool {
        matches!(character, ' ' | '\t')
    }

    fn is_printable_ascii(character: char) -> bool {
        matches!(character, '!'..='~')
    }
}

/// Two adjacent characters of a text, between which a line may break.
struct CharacterPair {
    before_previous: Option<char>,
    previous: char,
    /// The line break class of the last character before `character` that is not a combining
    /// mark.
    previous_class: LineBreak,
    character: char,
    class: LineBreak,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct LinePiecesCase {
        text: &'static str,
        word_break: WordBreak,
        pieces: &'static [&'static str],
    }

    fn pieces_at(text: &str, offsets: &[usize]) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut start = 0;
        for offset in offsets.iter().copied().chain([text.len()]) {
            pieces.push(text[start..offset].to_string());
            start = offset;
        }
        pieces
    }

    // The pieces are the lines Chromium lays the text out in when its box is too narrow for any
    // two pieces to share a line.
    #[test]
    fn test_line_start_offsets_match_chromium() {
        let cases = [
            LinePiecesCase {
                text: "2026년 11월 14일 오전 10:30 (Asia/Seoul)",
                word_break: WordBreak::Normal,
                pieces: &[
                    "2026",
                    "년 ",
                    "11",
                    "월 ",
                    "14",
                    "일 ",
                    "오",
                    "전 ",
                    "10:30 ",
                    "(Asia/Seoul)",
                ],
            },
            LinePiecesCase {
                text: "2026년 11월 14일 오전 10:30 (Asia/Seoul)",
                word_break: WordBreak::KeepAll,
                pieces: &[
                    "2026년 ",
                    "11월 ",
                    "14일 ",
                    "오전 ",
                    "10:30 ",
                    "(Asia/Seoul)",
                ],
            },
            LinePiecesCase {
                text: "2026년 11월 14일 오전 10:30 (Asia/Seoul)",
                word_break: WordBreak::BreakAll,
                pieces: &[
                    "2", "0", "2", "6", "년 ", "1", "1", "월 ", "1", "4", "일 ", "오", "전 ", "1",
                    "0:", "3", "0 ", "(A", "s", "i", "a/", "S", "e", "o", "u", "l)",
                ],
            },
            LinePiecesCase {
                text: "강의 (스캔).pdf",
                word_break: WordBreak::Normal,
                pieces: &["강", "의 ", "(스", "캔).pdf"],
            },
            LinePiecesCase {
                text: "강의 (스캔).pdf",
                word_break: WordBreak::KeepAll,
                pieces: &["강의 ", "(스캔).pdf"],
            },
            LinePiecesCase {
                text: "파일: 문서/보고서_최종(2).docx",
                word_break: WordBreak::KeepAll,
                pieces: &["파일: ", "문서/", "보고서", "_", "최종", "(2).docx"],
            },
            LinePiecesCase {
                text: "a/b c?d=e https://example.com/path-to/file_name.txt",
                word_break: WordBreak::KeepAll,
                pieces: &[
                    "a/b ",
                    "c?",
                    "d=e ",
                    "https://example.com/path-",
                    "to/file_name.txt",
                ],
            },
            LinePiecesCase {
                text: "a/b c?d=e https://example.com/path-to/file_name.txt",
                word_break: WordBreak::BreakAll,
                pieces: &[
                    "a/", "b ", "c?", "d", "=", "e ", "h", "t", "t", "p", "s://", "e", "x", "a",
                    "m", "p", "l", "e.", "c", "o", "m/", "p", "a", "t", "h", "-", "t", "o/", "f",
                    "i", "l", "e", "_", "n", "a", "m", "e.", "t", "x", "t",
                ],
            },
            LinePiecesCase {
                text: "x86_64-unknown-linux-gnu v1.2.3-beta+build.5",
                word_break: WordBreak::KeepAll,
                pieces: &[
                    "x86_64-",
                    "unknown-",
                    "linux-",
                    "gnu ",
                    "v1.2.3-",
                    "beta+build.5",
                ],
            },
            LinePiecesCase {
                text: "10,000원 3–5개 #태그 @사람",
                word_break: WordBreak::Normal,
                pieces: &[
                    "10,000", "원 ", "3–", "5", "개 ", "#", "태", "그 ", "@", "사", "람",
                ],
            },
            LinePiecesCase {
                text: "10,000원 3–5개 #태그 @사람",
                word_break: WordBreak::KeepAll,
                pieces: &["10,000원 ", "3–", "5개 ", "#", "태그 ", "@", "사람"],
            },
            LinePiecesCase {
                text: "“인용”과 ‘작은따옴표’로",
                word_break: WordBreak::KeepAll,
                pieces: &["“인용”", "과 ", "‘작은따옴표’", "로"],
            },
            LinePiecesCase {
                text: "日本語のテキスト、です。",
                word_break: WordBreak::KeepAll,
                pieces: &["日本語のテキスト、", "です。"],
            },
            LinePiecesCase {
                text: "Hello, world! (test) [x] {y}",
                word_break: WordBreak::BreakAll,
                pieces: &[
                    "H", "e", "l", "l", "o, ", "w", "o", "r", "l", "d! ", "(t", "e", "s", "t) ",
                    "[x] ", "{y}",
                ],
            },
        ];
        for case in cases {
            let line_breaking = LineBreaking {
                word_break: case.word_break,
                overflow_wrap: OverflowWrap::default(),
            };
            let offsets = line_breaking.line_start_offsets(case.text).unwrap();
            assert_eq!(pieces_at(case.text, &offsets), case.pieces, "{case:?}");
        }
    }

    #[test]
    fn test_word_characters_leave_opportunities_to_the_wrapper() {
        assert_eq!(LineBreaking::default().line_start_offsets("a b"), None);
    }
}
