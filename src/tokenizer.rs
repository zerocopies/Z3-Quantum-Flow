// tokenizer.rs — pure Rust BPE tokenizer for Llama 3.1
//
// Reads vocabulary directly from the already-parsed GgufMeta structure
// (which itself points into the mmap region). Zero heap copies of vocab
// strings beyond what's needed for the merge-priority HashMap.
//
// Implements:
//   - encode(text) → Vec<u32>   — BPE tokenisation with byte-fallback
//   - decode(ids)  → String     — id sequence back to UTF-8 text
//   - Special tokens: BOS (128000), EOS (128001), PAD (128004)
//
// Does NOT require llama.cpp. Does NOT require sentencepiece.
// The only input is the GgufMeta loaded by gguf.rs.

use std::collections::HashMap;
use std::fmt;

// ── Constants for Llama 3.1 ───────────────────────────────────────────────────

pub const TOKEN_BOS: u32 = 128_000;
pub const TOKEN_EOS: u32 = 128_001;
pub const TOKEN_EOT: u32 = 128_009; // <|eot_id|> — Llama 3.1 instruct stop token
pub const TOKEN_PAD: u32 = 128_004;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum TokenizerError {
    MissingVocab,
    MissingMerges,
    VocabSizeMismatch { tokens: usize, scores: usize },
    InvalidUtf8(std::string::FromUtf8Error),
    UnknownToken(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingVocab  => write!(f, "GGUF metadata missing tokenizer.ggml.tokens"),
            Self::MissingMerges => write!(f, "GGUF metadata missing tokenizer.ggml.merges"),
            Self::VocabSizeMismatch { tokens, scores } =>
                write!(f, "vocab size mismatch: {tokens} tokens vs {scores} scores"),
            Self::InvalidUtf8(e) => write!(f, "UTF-8 decode error: {e}"),
            Self::UnknownToken(s) => write!(f, "unknown token: {s:?}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

// ── Vocabulary ────────────────────────────────────────────────────────────────

/// A single vocab entry.
#[derive(Debug, Clone)]
struct VocabEntry {
    text: Vec<u8>,   // raw bytes (may not be valid UTF-8 for byte tokens)
    _score: f32,
    token_type: u32, // 1=normal, 2=unknown, 3=control, 6=byte
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

pub struct Tokenizer {
    /// id → entry
    vocab: Vec<VocabEntry>,
    /// text (UTF-8 or byte repr) → id  — for encode lookup
    token_to_id: HashMap<Vec<u8>, u32>,
    /// BPE merge priorities: (left_id, right_id) → rank (lower = higher priority)
    merge_rank: HashMap<(u32, u32), u32>,
    /// Pre-built: byte value (0-255) → token id for byte-fallback
    byte_to_token: [u32; 256],
}

impl Tokenizer {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Build a Tokenizer from raw string slices extracted from GGUF metadata.
    ///
    /// `tokens`  — `tokenizer.ggml.tokens`  (one entry per vocab id)
    /// `scores`  — `tokenizer.ggml.scores`  (float32, same length)
    /// `types`   — `tokenizer.ggml.token_type` (u32, same length; may be empty → all normal)
    /// `merges`  — `tokenizer.ggml.merges`  (one "A B" string per merge rule)
    pub fn from_gguf_parts(
        tokens: &[String],
        scores: &[f32],
        types: &[u32],
        merges: &[String],
    ) -> Result<Self, TokenizerError> {
        let n = tokens.len();
        if !scores.is_empty() && scores.len() != n {
            return Err(TokenizerError::VocabSizeMismatch { tokens: n, scores: scores.len() });
        }

        // Build vocab vec
        let mut vocab: Vec<VocabEntry> = Vec::with_capacity(n);
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(n);

        for (i, tok_str) in tokens.iter().enumerate() {
            let score = if !scores.is_empty() && i < scores.len() { scores[i] } else { 0.0 };
            let token_type = if i < types.len() { types[i] } else { 1 };

            // Llama 3.1 uses GPT-style byte tokens like <0x41> for 'A'
            let text: Vec<u8> = if let Some(byte_val) = parse_byte_token(tok_str) {
                vec![byte_val]
            } else {
                tok_str.as_bytes().to_vec()
            };

            token_to_id.insert(text.clone(), i as u32);
            vocab.push(VocabEntry { text, _score: score, token_type });
        }

        // Build merge rank table
        let mut merge_rank: HashMap<(u32, u32), u32> = HashMap::with_capacity(merges.len());
        for (rank, merge_str) in merges.iter().enumerate() {
            // Each merge entry is two space-separated token strings
            if let Some((left_str, right_str)) = merge_str.split_once(' ') {
                let left_bytes = left_str.as_bytes().to_vec();
                let right_bytes = right_str.as_bytes().to_vec();
                if let (Some(&lid), Some(&rid)) =
                    (token_to_id.get(&left_bytes), token_to_id.get(&right_bytes))
                {
                    merge_rank.insert((lid, rid), rank as u32);
                }
            }
        }

        // Build byte → token id table (for fallback on unknown characters)
        let mut byte_to_token = [u32::MAX; 256];
        for (id, entry) in vocab.iter().enumerate() {
            // Byte tokens are length-1 entries with token_type 6, OR <0xNN> tokens
            if entry.token_type == 6 || (entry.text.len() == 1 && entry.token_type != 1) {
                byte_to_token[entry.text[0] as usize] = id as u32;
            }
        }
        // Fill any gaps using the <0xNN> lookup as backup
        for byte_val in 0u8..=255 {
            if byte_to_token[byte_val as usize] == u32::MAX {
                let repr = format!("<0x{:02X}>", byte_val);
                if let Some(&id) = token_to_id.get(repr.as_bytes()) {
                    byte_to_token[byte_val as usize] = id;
                } else {
                    // Last resort: 0 (unknown token id in most vocabs)
                    byte_to_token[byte_val as usize] = 0;
                }
            }
        }

        Ok(Self { vocab, token_to_id, merge_rank, byte_to_token })
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    // ── Encode ───────────────────────────────────────────────────────────────

    /// Encode `text` to a token id sequence.
    ///
    /// Prepends BOS automatically (Llama 3.1 chat format requires it).
    /// Uses GPT-2/tiktoken-style BPE with byte-fallback for unknown characters.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        if add_bos { ids.push(TOKEN_BOS); }

        // Initial segmentation: each UTF-8 char → token id(s) via vocab lookup
        // or byte fallback
        let mut working: Vec<u32> = Vec::with_capacity(text.len());
        self.initial_tokenise(text, &mut working);

        // BPE merge loop
        self.bpe_merge(&mut working);

        ids.extend_from_slice(&working);
        ids
    }

    /// Encode without BOS (used internally for continuation tokens).
    pub fn encode_no_bos(&self, text: &str) -> Vec<u32> {
        self.encode(text, false)
    }

    // ── Decode ───────────────────────────────────────────────────────────────

    /// Decode a sequence of token ids back to a UTF-8 String.
    /// Skips BOS/EOS/PAD and other control tokens silently.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            // Skip special tokens
            if id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_PAD { continue; }
            if let Some(entry) = self.vocab.get(id as usize) {
                // Control tokens (type 3) → skip
                if entry.token_type == 3 { continue; }
                bytes.extend_from_slice(&entry.text);
            }
        }
        // Convert bytes to UTF-8, replacing any invalid sequences with '?'
        String::from_utf8(bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    /// Decode a single token id to its string representation.
    /// Useful for streaming output token-by-token.
    pub fn decode_one(&self, id: u32) -> Option<String> {
        if id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_EOT || id == TOKEN_PAD { return None; }
        let entry = self.vocab.get(id as usize)?;
        if entry.token_type == 3 { return None; }
        let s = String::from_utf8_lossy(&entry.text).into_owned();
        // GPT-2 style: 'Ġ' (U+0120) represents a leading space
        Some(s.replace('\u{0120}', " ").replace('\u{010a}', "\n"))
    }

    pub fn is_eos(&self, id: u32) -> bool { id == TOKEN_EOS || id == TOKEN_EOT }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Convert each character of `text` to its initial token id(s).
    /// Uses direct vocab lookup first; falls back to byte tokens.
    ///
    /// Input text is first converted to GPT-2's byte-to-unicode representation
    /// (space → Ġ, newline → Ċ, etc.) so that lookups match the vocab storage format.
    fn initial_tokenise(&self, text: &str, out: &mut Vec<u32>) {
        // Convert input bytes to GPT-2 unicode encoding before lookup.
        // The vocab stores tokens as GPT-2 unicode (e.g. " is" → "Ġis"),
        // so we must convert the raw input bytes to match.
        let gpt2_bytes = bytes_to_gpt2_unicode(text.as_bytes());
        let bytes = &gpt2_bytes;
        let mut i = 0;
        while i < bytes.len() {
            let mut matched = false;
            let max_len = (bytes.len() - i).min(64);
            for len in (1..=max_len).rev() {
                let slice = &bytes[i..i + len];
                if let Some(&id) = self.token_to_id.get(slice) {
                    out.push(id);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Byte fallback: use the original input byte, not the unicode repr
                // Find the original byte at this position by scanning back
                out.push(self.byte_to_token[orig_byte_at(text.as_bytes(), bytes, i)]);
                i += gpt2_char_len(bytes[i]);
            }
        }
    }

    /// BPE merge pass: repeatedly merge the highest-priority (lowest-rank) pair.
    fn bpe_merge(&self, ids: &mut Vec<u32>) {
        loop {
            if ids.len() < 2 { break; }

            // Find the best merge (lowest rank = highest priority)
            let mut best_rank = u32::MAX;
            let mut best_pos = usize::MAX;

            for i in 0..ids.len() - 1 {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                    }
                }
            }

            if best_pos == usize::MAX { break; } // no more merges possible

            // Look up merged token id: concatenate the two token strings
            let left_text = &self.vocab[ids[best_pos] as usize].text;
            let right_text = &self.vocab[ids[best_pos + 1] as usize].text;
            let mut merged_text = left_text.clone();
            merged_text.extend_from_slice(right_text);

            if let Some(&merged_id) = self.token_to_id.get(&merged_text) {
                ids[best_pos] = merged_id;
                ids.remove(best_pos + 1);
            } else {
                // Merge string not found in vocab — this shouldn't happen with a
                // correct merge table, but protect against it rather than panicking.
                break;
            }
        }
    }
}

// ── Helper: parse <0xNN> byte token notation ──────────────────────────────────

fn parse_byte_token(s: &str) -> Option<u8> {
    // Matches "<0x41>" style tokens used in Llama / GPT-J vocabs
    let s = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(s, 16).ok()
}

// ── GPT-2 byte-to-unicode helpers ─────────────────────────────────────────────

/// Convert raw input bytes to their GPT-2 unicode representation as UTF-8.
///
/// GPT-2 maps each byte to a unicode character:
/// - Printable ASCII (33–126), Latin-1 supplement (161–172, 174–255) → themselves
/// - Everything else (0–32, 127–160, 173) → U+0100..U+015F
///
/// This is needed because the Llama/GPT-2 BPE vocabulary stores tokens using
/// this encoding (e.g. space = Ġ = U+0120, newline = Ċ = U+010A).
fn bytes_to_gpt2_unicode(input: &[u8]) -> Vec<u8> {
    // Precomputed: each byte → its GPT-2 unicode codepoint
    // Bytes 33–126 → themselves (printable ASCII minus space)
    // Bytes 161–172, 174–255 → themselves (printable Latin-1)
    // Everything else → U+0100 + offset (non-printable bytes)
    static TABLE: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut n = 0u32;
        let mut b = 0usize;
        while b < 256 {
            let is_printable = (b >= 33 && b <= 126)
                || (b >= 161 && b <= 172)
                || (b >= 174 && b <= 255);
            if is_printable {
                t[b] = b as u32;
            } else {
                t[b] = 256 + n;
                n += 1;
            }
            b += 1;
        }
        t
    };

    let mut out = Vec::with_capacity(input.len() * 2);
    for &byte in input {
        let cp = TABLE[byte as usize];
        // Encode codepoint as UTF-8
        if cp < 0x80 {
            out.push(cp as u8);
        } else if cp < 0x800 {
            out.push(0xC0 | (cp >> 6) as u8);
            out.push(0x80 | (cp & 0x3F) as u8);
        } else {
            out.push(0xE0 | (cp >> 12) as u8);
            out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
            out.push(0x80 | (cp & 0x3F) as u8);
        }
    }
    out
}

/// Get the number of UTF-8 bytes used by the GPT-2 unicode char starting at `gpt2[i]`.
#[inline]
fn gpt2_char_len(first_byte: u8) -> usize {
    if first_byte < 0x80 { 1 }
    else if first_byte < 0xE0 { 2 }
    else { 3 }
}

/// Map position `i` in the gpt2-encoded byte slice back to the original input byte.
/// Since each input byte maps to exactly one GPT-2 character, we scan forward.
fn orig_byte_at(orig: &[u8], gpt2: &[u8], gpt2_pos: usize) -> usize {
    let mut g = 0usize;
    for (orig_idx, _) in orig.iter().enumerate() {
        if g == gpt2_pos { return orig_idx; }
        g += gpt2_char_len(gpt2[g]);
    }
    0
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_tokenizer() -> Tokenizer {
        // Tiny vocab: " ", "h", "e", "l", "o", "he", "hel", "hell", "hello"
        let tokens = vec![
            "<unk>".to_string(),   // 0
            "h".to_string(),       // 1
            "e".to_string(),       // 2
            "l".to_string(),       // 3
            "o".to_string(),       // 4
            "he".to_string(),      // 5
            "hel".to_string(),     // 6
            "hell".to_string(),    // 7
            "hello".to_string(),   // 8
            " ".to_string(),       // 9
        ];
        let scores: Vec<f32> = vec![0.0; tokens.len()];
        let types: Vec<u32> = vec![2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        // merges: h+e → he (rank 0), he+l → hel (rank 1), hel+l → hell (rank 2), hell+o → hello (rank 3)
        let merges = vec![
            "h e".to_string(),
            "he l".to_string(),
            "hel l".to_string(),
            "hell o".to_string(),
        ];
        Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges).unwrap()
    }

    #[test]
    fn encode_hello() {
        let t = minimal_tokenizer();
        let ids = t.encode_no_bos("hello");
        assert_eq!(ids, vec![8]); // "hello" should merge to a single token
    }

    #[test]
    fn decode_roundtrip() {
        let t = minimal_tokenizer();
        let ids = t.encode_no_bos("hello");
        let text = t.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn bos_prepend() {
        let t = minimal_tokenizer();
        let ids = t.encode("hello", true);
        assert_eq!(ids[0], TOKEN_BOS);
    }

    #[test]
    fn byte_token_parse() {
        assert_eq!(parse_byte_token("<0x41>"), Some(b'A'));
        assert_eq!(parse_byte_token("<0x00>"), Some(0));
        assert_eq!(parse_byte_token("hello"), None);
    }
}
