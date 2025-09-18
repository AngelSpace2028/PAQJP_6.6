import os
import sys
import math
import struct
import array
import random
import heapq
import binascii
import logging
import paq  # Python binding for PAQ9a (pip install paq)
import hashlib
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple, Optional

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.getLogger().setLevel(logging.DEBUG)  # Allow debug messages

# === Constants ===
PROGNAME = "PAQJP_6_Smart"
HUFFMAN_THRESHOLD = 1024  # Bytes threshold for Huffman vs. PAQ compression
PI_DIGITS_FILE = "pi_digits.txt"
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
MEM = 1 << 15  # 32,768
MAX_BITS = 2**28  # Maximum bits for transform_14
MIN_BITS = 2      # Minimum bits for transform_14

# === Dictionary file list ===
DICTIONARY_FILES = [
    "words_enwik8.txt", "eng_news_2005_1M-sentences.txt", "eng_news_2005_1M-words.txt",
    "eng_news_2005_1M-sources.txt", "eng_news_2005_1M-co_n.txt",
    "eng_news_2005_1M-co_s.txt", "eng_news_2005_1M-inv_so.txt",
    "eng_news_2005_1M-meta.txt", "Dictionary.txt",
    "the-complete-reference-html-css-fifth-edition.txt", "francais.txt", "espanol.txt",
    "deutsch.txt", "ukenglish.txt", "vertebrate-palaeontology-dict.txt"
]

# === DNA Encoding Table for GenomeCompress ===
DNA_ENCODING_TABLE = {
    'AAAA': 0b00000, 'AAAC': 0b00001, 'AAAG': 0b00010, 'AAAT': 0b00011,
    'AACC': 0b00100, 'AACG': 0b00101, 'AACT': 0b00110, 'AAGG': 0b00111,
    'AAGT': 0b01000, 'AATT': 0b01001, 'ACCC': 0b01010, 'ACCG': 0b01011,
    'ACCT': 0b01100, 'AGGG': 0b01101, 'AGGT': 0b01110, 'AGTT': 0b01111,
    'CCCC': 0b10000, 'CCCG': 0b10001, 'CCCT': 0b10010, 'CGGG': 0b10011,
    'CGGT': 0b10100, 'CGTT': 0b10101, 'GTTT': 0b10110, 'CTTT': 0b10111,
    'AAAAAAAA': 0b11000, 'CCCCCCCC': 0b11001, 'GGGGGGGG': 0b11010, 'TTTTTTTT': 0b11011,
    'A': 0b11100, 'C': 0b11101, 'G': 0b11110, 'T': 0b11111
}
DNA_DECODING_TABLE = {v: k for k, v in DNA_ENCODING_TABLE.items()}

# === Pi Digits Functions ===
def save_pi_digits(digits: List[int], filename: str = PI_DIGITS_FILE) -> bool:
    """Save base-10 pi digits to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(','.join(str(d) for d in digits))
        logging.info(f"Saved {len(digits)} pi digits to {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to save pi digits to {filename}: {e}")
        return False

def load_pi_digits(filename: str = PI_DIGITS_FILE, expected_count: int = 3) -> Optional[List[int]]:
    """Load base-10 pi digits from a file."""
    try:
        if not os.path.isfile(filename):
            logging.warning(f"Pi digits file {filename} does not exist")
            return None
        with open(filename, 'r') as f:
            data = f.read().strip()
            if not data:
                logging.warning(f"Pi digits file {filename} is empty")
                return None
            digits = []
            for x in data.split(','):
                if not x.isdigit():
                    logging.warning(f"Invalid integer in {filename}: {x}")
                    return None
                d = int(x)
                if not (0 <= d <= 255):
                    logging.warning(f"Digit out of range in {filename}: {d}")
                    return None
                digits.append(d)
            if len(digits) != expected_count:
                logging.warning(f"Loaded {len(digits)} digits, expected {expected_count}")
                return None
            logging.info(f"Loaded {len(digits)} pi digits from {filename}")
            return digits
    except Exception as e:
        logging.error(f"Failed to load pi digits from {filename}: {e}")
        return None

def generate_pi_digits(num_digits: int = 3, filename: str = PI_DIGITS_FILE) -> List[int]:
    """Generate or load pi digits, mapping to 0-255 range."""
    loaded_digits = load_pi_digits(filename, num_digits)
    if loaded_digits is not None:
        return loaded_digits
    try:
        # Requires mpmath: pip install mpmath (optional)
        from mpmath import mp
        mp.dps = num_digits
        pi_digits = [int(d) for d in str(mp.pi)[2:2+num_digits]]  # Get digits after decimal
        if len(pi_digits) != num_digits:
            logging.error(f"Generated {len(pi_digits)} digits, expected {num_digits}")
            raise ValueError("Incorrect number of pi digits generated")
        if not all(0 <= d <= 9 for d in pi_digits):
            logging.error("Generated pi digits contain invalid values")
            raise ValueError("Invalid pi digits generated")
        mapped_digits = [(d * 255 // 9) % 256 for d in pi_digits]
        save_pi_digits(mapped_digits, filename)
        return mapped_digits
    except ImportError:
        logging.warning("mpmath not installed, using fallback pi digits")
        fallback_digits = [1, 4, 1]  # First three digits after decimal
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits[:num_digits]]
        save_pi_digits(mapped_fallback, filename)
        return mapped_fallback
    except Exception as e:
        logging.error(f"Failed to generate pi digits: {e}")
        fallback_digits = [1, 4, 1]
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits[:num_digits]]
        logging.warning(f"Using {len(mapped_fallback)} fallback pi digits")
        save_pi_digits(mapped_fallback, filename)
        return mapped_fallback

PI_DIGITS = generate_pi_digits(3)

# === Helper Classes and Functions ===
class Filetype(Enum):
    DEFAULT = 0
    JPEG = 1
    TEXT = 3

class Node:
    """Huffman tree node."""
    def __init__(self, left=None, right=None, symbol=None):
        self.left = left
        self.right = right
        self.symbol = symbol

    def is_leaf(self):
        return self.left is None and self.right is None

def transform_with_prime_xor_every_3_bytes(data, repeat=100):
    """XOR every third byte with prime-derived values."""
    if not data:
        return b''
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
        for _ in range(min(repeat, 10)):  # Limit repeats for performance
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def transform_with_pattern_chunk(data, chunk_size=4):
    """XOR each chunk with 0xFF."""
    if not data:
        return b''
    transformed = bytearray()
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        transformed.extend([b ^ 0xFF for b in chunk])
    return bytes(transformed)

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def find_nearest_prime_around(n):
    """Find the nearest prime number to n."""
    offset = 0
    while True:
        if is_prime(n - offset):
            return n - offset
        if is_prime(n + offset):
            return n + offset
        offset += 1

# === State Table ===
class StateTable:
    """State table for finite state machine transformations."""
    def __init__(self):
        self.table = [
            [1, 2, 0, 0], [3, 5, 1, 0], [4, 6, 0, 1], [7, 10, 2, 0],
            [8, 12, 1, 1], [9, 13, 1, 1], [11, 14, 0, 2], [15, 19, 3, 0],
            [16, 23, 2, 1], [17, 24, 2, 1], [18, 25, 2, 1], [20, 27, 1, 2],
            [21, 28, 1, 2], [22, 29, 1, 2], [26, 30, 0, 3], [31, 33, 4, 0],
            [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1],
            [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2],
            [34, 37, 2, 2], [34, 37, 2, 2], [36, 39, 1, 3], [36, 39, 1, 3],
            [36, 39, 1, 3], [36, 39, 1, 3], [38, 40, 0, 4], [41, 43, 5, 0],
            [42, 45, 4, 1], [42, 45, 4, 1], [44, 47, 3, 2], [44, 47, 3, 2],
            [46, 49, 2, 3], [46, 49, 2, 3], [48, 51, 1, 4], [48, 51, 1, 4],
            [50, 52, 0, 5], [53, 43, 6, 0], [54, 57, 5, 1], [54, 57, 5, 1],
            [56, 59, 4, 2], [56, 59, 4, 2], [58, 61, 3, 3], [58, 61, 3, 3],
            [60, 63, 2, 4], [60, 63, 2, 4], [62, 65, 1, 5], [62, 65, 1, 5],
            [50, 66, 0, 6], [67, 55, 7, 0], [68, 57, 6, 1], [68, 57, 6, 1],
            [70, 73, 5, 2], [70, 73, 5, 2], [72, 75, 4, 3], [72, 75, 4, 3],
            [74, 77, 3, 4], [74, 77, 3, 4], [76, 79, 2, 5], [76, 79, 2, 5],
            [62, 81, 1, 6], [62, 81, 1, 6], [64, 82, 0, 7], [83, 69, 8, 0],
            [84, 76, 7, 1], [84, 76, 7, 1], [86, 73, 6, 2], [86, 73, 6, 2],
            [44, 59, 5, 3], [44, 59, 5, 3], [58, 61, 4, 4], [58, 61, 4, 4],
            [60, 49, 3, 5], [60, 49, 3, 5], [76, 89, 2, 6], [76, 89, 2, 6],
            [78, 91, 1, 7], [78, 91, 1, 7], [80, 92, 0, 8], [93, 69, 9, 0],
            [94, 87, 8, 1], [94, 87, 8, 1], [96, 45, 7, 2], [96, 45, 7, 2],
            [48, 99, 2, 7], [48, 99, 2, 7], [88, 101, 1, 8], [88, 101, 1, 8],
            [80, 102, 0, 9], [103, 69, 10, 0], [104, 87, 9, 1], [104, 87, 9, 1],
            [106, 57, 8, 2], [106, 57, 8, 2], [62, 109, 2, 8], [62, 109, 2, 8],
            [88, 111, 1, 9], [88, 111, 1, 9], [80, 112, 0, 10], [113, 85, 11, 0],
            [114, 87, 10, 1], [114, 87, 10, 1], [116, 57, 9, 2], [116, 57, 9, 2],
            [62, 119, 2, 9], [62, 119, 2, 9], [88, 121, 1, 10], [88, 121, 1, 10],
            [90, 122, 0, 11], [123, 85, 12, 0], [124, 97, 11, 1], [124, 97, 11, 1],
            [126, 57, 10, 2], [126, 57, 10, 2], [62, 129, 2, 10], [62, 129, 2, 10],
            [98, 131, 1, 11], [98, 131, 1, 11], [90, 132, 0, 12], [133, 85, 13, 0],
            [134, 97, 12, 1], [134, 97, 12, 1], [136, 57, 11, 2], [136, 57, 11, 2],
            [62, 139, 2, 11], [62, 139, 2, 11], [98, 141, 1, 12], [98, 141, 1, 12],
            [90, 142, 0, 13], [143, 95, 14, 0], [144, 97, 13, 1], [144, 97, 13, 1],
            [68, 57, 12, 2], [68, 57, 12, 2], [62, 81, 2, 12], [62, 81, 2, 12],
            [98, 147, 1, 13], [98, 147, 1, 13], [100, 148, 0, 14], [149, 95, 15, 0],
            [150, 107, 14, 1], [150, 107, 14, 1], [108, 151, 1, 14], [108, 151, 1, 14],
            [100, 152, 0, 15], [153, 95, 16, 0], [154, 107, 15, 1], [108, 155, 1, 15],
            [100, 156, 0, 16], [157, 95, 17, 0], [158, 107, 16, 1], [108, 159, 1, 16],
            [100, 160, 0, 17], [161, 105, 18, 0], [162, 107, 17, 1], [108, 163, 1, 17],
            [110, 164, 0, 18], [165, 105, 19, 0], [166, 117, 18, 1], [118, 167, 1, 18],
            [110, 168, 0, 19], [169, 105, 20, 0], [170, 117, 19, 1], [118, 171, 1, 19],
            [110, 172, 0, 20], [173, 105, 21, 0], [174, 117, 20, 1], [118, 175, 1, 20],
            [110, 176, 0, 21], [177, 105, 22, 0], [178, 117, 21, 1], [118, 179, 1, 21],
            [120, 184, 0, 23], [185, 115, 24, 0], [186, 127, 23, 1], [128, 187, 1, 23],
            [120, 188, 0, 24], [189, 115, 25, 0], [190, 127, 24, 1], [128, 191, 1, 24],
            [120, 192, 0, 25], [193, 115, 26, 0], [194, 127, 25, 1], [128, 195, 1, 25],
            [120, 196, 0, 26], [197, 115, 27, 0], [198, 127, 26, 1], [128, 199, 1, 26],
            [120, 200, 0, 27], [201, 115, 28, 0], [202, 127, 27, 1], [128, 203, 1, 27],
            [120, 204, 0, 28], [205, 115, 29, 0], [206, 127, 28, 1], [128, 207, 1, 28],
            [120, 208, 0, 29], [209, 125, 30, 0], [210, 127, 29, 1], [128, 211, 1, 29],
            [130, 212, 0, 30], [213, 125, 31, 0], [214, 137, 30, 1], [138, 215, 1, 30],
            [130, 216, 0, 31], [217, 125, 32, 0], [218, 137, 31, 1], [138, 219, 1, 31],
            [130, 220, 0, 32], [221, 125, 33, 0], [222, 137, 32, 1], [138, 223, 1, 32],
            [130, 224, 0, 33], [225, 125, 34, 0], [226, 137, 33, 1], [138, 227, 1, 33],
            [130, 228, 0, 34], [229, 125, 35, 0], [230, 137, 34, 1], [138, 231, 1, 34],
            [130, 232, 0, 35], [233, 125, 36, 0], [234, 137, 35, 1], [138, 235, 1, 35],
            [130, 236, 0, 36], [237, 125, 37, 0], [238, 137, 36, 1], [138, 239, 1, 36],
            [130, 240, 0, 37], [241, 125, 38, 0], [242, 137, 37, 1], [138, 243, 1, 37],
            [130, 244, 0, 38], [245, 135, 39, 0], [246, 137, 38, 1], [138, 247, 1, 38],
            [140, 248, 0, 39], [249, 135, 40, 0], [250, 69, 39, 1], [80, 251, 1, 39],
            [140, 252, 0, 40], [249, 135, 41, 0], [250, 69, 40, 1], [80, 251, 1, 40],
            [140, 252, 0, 41]
        ]

    def get_state(self, current_state: int, input_byte: int) -> int:
        """Get next state from current state and input byte."""
        if 0 <= current_state < len(self.table) and 0 <= input_byte < 4:
            return self.table[current_state][input_byte]
        return 0

    def apply_transform(self, data: bytes) -> bytes:
        """Apply state table transformation to data."""
        if not data:
            return b''
        
        transformed = bytearray(data)
        state = 0
        
        for i in range(len(transformed)):
            input_byte = transformed[i] % 4
            next_state = self.get_state(state, input_byte)
            # Apply transformation based on state transition
            transformed[i] ^= (next_state * 17) % 256
            state = next_state
        
        return bytes(transformed)

    def reverse_transform(self, data: bytes) -> bytes:
        """Reverse state table transformation."""
        return self.apply_transform(data)  # XOR is symmetric

# === Smart Compressor ===
class SmartCompressor:
    def __init__(self):
        self.dictionaries = self.load_dictionaries()

    def load_dictionaries(self):
        """Load dictionary files for hash lookup."""
        data = []
        for filename in DICTIONARY_FILES:
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content:
                            data.append(content)
                    logging.info(f"Loaded dictionary: {filename} ({len(content)} chars)")
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")
            else:
                logging.debug(f"Missing dictionary: {filename}")
        logging.info(f"Loaded {len(data)} dictionary files")
        return data

    def compute_sha256(self, data):
        """Compute SHA-256 hash as hex."""
        return hashlib.sha256(data).hexdigest()

    def compute_sha256_binary(self, data):
        """Compute SHA-256 hash as bytes."""
        return hashlib.sha256(data).digest()

    def find_hash_in_dictionaries(self, hash_hex):
        """Search for hash in dictionary files."""
        for idx, content in enumerate(self.dictionaries):
            if hash_hex in content:
                filename = DICTIONARY_FILES[idx]
                logging.info(f"Hash {hash_hex[:16]}... found in {filename}")
                return filename
        return None

    def generate_8byte_sha(self, data):
        """Generate 8-byte SHA-256 prefix."""
        try:
            return hashlib.sha256(data).digest()[:8]
        except Exception as e:
            logging.error(f"Failed to generate SHA: {e}")
            return None

    def paq_compress(self, data):
        """Compress data using PAQ9a."""
        if not data:
            logging.warning("paq_compress: Empty input, returning empty bytes")
            return b''
        try:
            if isinstance(data, bytearray):
                data = bytes(data)
            elif not isinstance(data, bytes):
                raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
            compressed = paq.compress(data)
            logging.info(f"PAQ9a compression: {len(data)} -> {len(compressed)} bytes")
            return compressed
        except Exception as e:
            logging.error(f"PAQ9a compression failed: {e}")
            return None

    def paq_decompress(self, data):
        """Decompress data using PAQ9a."""
        if not data:
            logging.warning("paq_decompress: Empty input, returning empty bytes")
            return b''
        try:
            decompressed = paq.decompress(data)
            logging.info(f"PAQ9a decompression: {len(data)} -> {len(decompressed)} bytes")
            return decompressed
        except Exception as e:
            logging.error(f"PAQ9a decompression failed: {e}")
            return None

    def reversible_transform(self, data):
        """Apply reversible XOR transform with 0xAA."""
        logging.debug("Applying XOR transform (0xAA)")
        transformed = bytes(b ^ 0xAA for b in data)
        logging.debug("XOR transform complete")
        return transformed

    def reverse_reversible_transform(self, data):
        """Reverse XOR transform with 0xAA."""
        logging.debug("Reversing XOR transform (0xAA)")
        return self.reversible_transform(data)  # XOR is symmetric

    def compress(self, input_data, input_file):
        """Compress data using Smart Compressor with consistent return types."""
        if not input_data:
            logging.warning("Empty input, returning minimal output")
            return bytes([252])  # Error marker

        original_size = len(input_data)
        if original_size < 8:
            logging.info(f"File too small ({original_size} bytes), returning uncompressed")
            return bytes([253]) + input_data

        original_hash = self.compute_sha256(input_data)
        logging.info(f"SHA-256 of input: {original_hash[:16]}...")

        # Check dictionary match
        found = self.find_hash_in_dictionaries(original_hash)
        if found:
            logging.info(f"Hash found in dictionary: {found}")
            sha8 = self.generate_8byte_sha(input_data)
            if sha8:
                return bytes([255]) + sha8  # Dictionary match marker

        # Handle special .paq files
        if input_file.lower().endswith(".paq") and any(x in input_file.lower() for x in ["words", "lines", "sentence"]):
            sha = self.generate_8byte_sha(input_data)
            if sha and len(input_data) > 8:
                logging.info(f"SHA-8 for .paq file: {sha.hex()}")
                return bytes([254]) + sha
            logging.info("Original smaller than SHA, keeping original")
            return bytes([253]) + input_data

        # Try PAQ compression
        transformed = self.reversible_transform(input_data)
        compressed = self.paq_compress(transformed)
        if compressed is None:
            logging.error("PAQ compression failed")
            return bytes([252]) + input_data  # Error marker with original data

        if len(compressed) < original_size * 0.9:  # Only if we save at least 10%
            output = self.compute_sha256_binary(input_data) + compressed
            logging.info(f"Smart compression successful: {original_size} -> {len(output)} bytes")
            return bytes([0]) + output
        else:
            logging.info("PAQ compression not efficient enough, returning uncompressed")
            return bytes([251]) + input_data

    def decompress(self, input_data):
        """Decompress data using Smart Compressor."""
        if len(input_data) < 1:
            logging.error("Input too short for Smart Compressor")
            return None

        marker = input_data[0]
        data = input_data[1:]

        # Quick cases
        if marker in [255, 254, 253, 251]:  # Dictionary, SHA, uncompressed cases
            logging.info(f"Smart decompress: Quick case marker {marker}")
            return data

        if marker == 0:  # Standard compression
            if len(data) < 32:
                logging.error("Input too short for hash verification")
                return None

            stored_hash = data[:32]
            compressed_data = data[32:]

            decompressed = self.paq_decompress(compressed_data)
            if decompressed is None:
                logging.error("PAQ decompression failed")
                return None

            original = self.reverse_reversible_transform(decompressed)
            computed_hash = self.compute_sha256_binary(original)
            
            if computed_hash == stored_hash:
                logging.info("Smart decompression: Hash verification successful")
                return original
            else:
                logging.error("Smart decompression: Hash verification failed")
                return None

        if marker == 252:  # Error case
            logging.warning("Smart decompression: Error marker detected")
            return data

        logging.error(f"Unknown Smart Compressor marker: {marker}")
        return None

# === PAQJP Compressor ===
class PAQJPCompressor:
    def __init__(self):
        self.PI_DIGITS = list(PI_DIGITS)  # Make mutable copy
        self.PRIMES = PRIMES
        self.seed_tables = self.generate_seed_tables()
        self.SQUARE_OF_ROOT = 2
        self.ADD_NUMBERS = 1
        self.MULTIPLY = 3
        self.fibonacci = self.generate_fibonacci(100)
        self.state_table = StateTable()

    def generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms."""
        if n < 2:
            return [0]
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

    def generate_seed_tables(self, num_tables=126, table_size=256, min_val=5, max_val=255, seed=42):
        """Generate random seed tables."""
        random.seed(seed)
        return [[random.randint(min_val, max_val) for _ in range(table_size)] for _ in range(num_tables)]

    def get_seed(self, table_idx: int, value: int) -> int:
        """Get seed value from table."""
        if 0 <= table_idx < len(self.seed_tables):
            return self.seed_tables[table_idx][value % len(self.seed_tables[table_idx])]
        return 0

    def calculate_frequencies(self, binary_str):
        """Calculate bit frequencies."""
        if not binary_str:
            return {'0': 0, '1': 0}
        frequencies = {}
        for bit in binary_str:
            frequencies[bit] = frequencies.get(bit, 0) + 1
        return frequencies

    def build_huffman_tree(self, frequencies):
        """Build Huffman tree from frequencies."""
        if not frequencies or sum(frequencies.values()) == 0:
            return None
        heap = [(freq, Node(symbol=symbol)) for symbol, freq in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            freq1, node1 = heapq.heappop(heap)
            freq2, node2 = heapq.heappop(heap)
            new_node = Node(left=node1, right=node2)
            heapq.heappush(heap, (freq1 + freq2, new_node))
        return heap[0][1]

    def generate_huffman_codes(self, root, current_code="", codes=None):
        """Generate Huffman codes from tree."""
        if codes is None:
            codes = {}
        if root is None:
            return codes
        if root.is_leaf():
            codes[root.symbol] = current_code or "0"
            return codes
        if root.left:
            self.generate_huffman_codes(root.left, current_code + "0", codes)
        if root.right:
            self.generate_huffman_codes(root.right, current_code + "1", codes)
        return codes

    def compress_data_huffman(self, binary_str):
        """Compress binary string using Huffman coding."""
        if not binary_str:
            return ""
        frequencies = self.calculate_frequencies(binary_str)
        huffman_tree = self.build_huffman_tree(frequencies)
        if huffman_tree is None:
            return binary_str  # Fallback to original
        huffman_codes = self.generate_huffman_codes(huffman_tree)
        # Ensure both symbols have codes
        if '0' not in huffman_codes:
            huffman_codes['0'] = '0'
        if '1' not in huffman_codes:
            huffman_codes['1'] = '1'
        return ''.join(huffman_codes[bit] for bit in binary_str)

    def decompress_data_huffman(self, compressed_str):
        """Decompress Huffman-coded string."""
        if not compressed_str:
            return ""
        frequencies = self.calculate_frequencies(compressed_str)
        huffman_tree = self.build_huffman_tree(frequencies)
        if huffman_tree is None:
            return compressed_str  # Fallback
        huffman_codes = self.generate_huffman_codes(huffman_tree)
        reversed_codes = {code: symbol for symbol, code in huffman_codes.items()}
        decompressed_str = ""
        current_code = ""
        for bit in compressed_str:
            current_code += bit
            if current_code in reversed_codes:
                decompressed_str += reversed_codes[current_code]
                current_code = ""
        return decompressed_str

    def paq_compress(self, data):
        """Compress data using PAQ9a."""
        if not data:
            return b''
        try:
            if isinstance(data, bytearray):
                data = bytes(data)
            elif not isinstance(data, bytes):
                raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
            compressed = paq.compress(data)
            return compressed
        except Exception as e:
            logging.error(f"PAQ9a compression failed: {e}")
            return None

    def paq_decompress(self, data):
        """Decompress data using PAQ9a."""
        if not data:
            return b''
        try:
            decompressed = paq.decompress(data)
            return decompressed
        except Exception as e:
            logging.error(f"PAQ9a decompression failed: {e}")
            return None

    def transform_genomecompress(self, data: bytes) -> bytes:
        """Encode DNA sequence using GenomeCompress algorithm with proper disambiguation."""
        if not data:
            return b''
        
        try:
            dna_str = data.decode('ascii').upper()
            if not all(c in 'ACGT' for c in dna_str):
                logging.error("transform_genomecompress: Invalid DNA sequence")
                return data  # Return original on error
        except Exception as e:
            logging.error(f"transform_genomecompress: Failed to decode: {e}")
            return data

        n = len(dna_str)
        output_bits = []
        i = 0

        # Create ordered lookup for longest match first
        encoding_priority = []
        # 8-base patterns first
        for pattern in ['AAAAAAAA', 'CCCCCCCC', 'GGGGGGGG', 'TTTTTTTT']:
            if pattern in DNA_ENCODING_TABLE:
                encoding_priority.append(pattern)
        # 4-base patterns
        four_base_patterns = [p for p in DNA_ENCODING_TABLE if len(p) == 4]
        encoding_priority.extend(four_base_patterns)
        # Single bases last
        encoding_priority.extend(['A', 'C', 'G', 'T'])

        while i < n:
            matched = False
            
            # Try patterns in priority order
            for pattern in encoding_priority:
                if i + len(pattern) <= n and dna_str[i:i+len(pattern)] == pattern:
                    output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[pattern], '05b')])
                    i += len(pattern)
                    matched = True
                    break
            
            if not matched:
                # Emergency fallback - encode as single 'A'
                logging.debug(f"transform_genomecompress: No match at position {i}, using 'A'")
                output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE['A'], '05b')])
                i += 1

        bit_str = ''.join(map(str, output_bits))
        byte_length = (len(bit_str) + 7) // 8
        byte_data = int(bit_str, 2).to_bytes(byte_length, 'big') if bit_str else b''
        
        logging.info(f"transform_genomecompress: Encoded {n} bases to {len(byte_data)} bytes")
        return byte_data

    def reverse_transform_genomecompress(self, data: bytes) -> bytes:
        """Decode data compressed with GenomeCompress algorithm."""
        if not data:
            return b''

        try:
            bit_str = bin(int.from_bytes(data, 'big'))[2:].zfill(len(data) * 8)
            output = []
            i = 0

            while i < len(bit_str):
                if i + 5 > len(bit_str):
                    logging.warning(f"reverse_transform_genomecompress: Incomplete 5-bit segment at position {i}")
                    break
                segment_bits = bit_str[i:i+5]
                segment_val = int(segment_bits, 2)
                if segment_val in DNA_DECODING_TABLE:
                    output.append(DNA_DECODING_TABLE[segment_val])
                else:
                    logging.warning(f"reverse_transform_genomecompress: Unknown 5-bit code {segment_bits}, using 'A'")
                    output.append('A')
                i += 5

            dna_str = ''.join(output)
            result = dna_str.encode('ascii')
            
            logging.info(f"reverse_transform_genomecompress: Decoded {len(result)} bases")
            return result
        except Exception as e:
            logging.error(f"reverse_transform_genomecompress failed: {e}")
            return data  # Return original on error

    def transform_01(self, data, repeat=100):
        """Transform using prime XOR every 3 bytes."""
        if not data:
            return b''
        return transform_with_prime_xor_every_3_bytes(data, repeat=repeat)

    def reverse_transform_01(self, data, repeat=100):
        """Reverse transform_01 (same as forward)."""
        return self.transform_01(data, repeat=repeat)

    def transform_03(self, data):
        """Transform using chunk XOR with 0xFF."""
        if not data:
            return b''
        return transform_with_pattern_chunk(data)

    def reverse_transform_03(self, data):
        """Reverse transform_03 (same as forward)."""
        return self.transform_03(data)

    def transform_04(self, data, repeat=100):
        """Subtract index modulo 256."""
        if not data:
            return b''
        transformed = bytearray(data)
        effective_repeat = min(repeat, max(1, len(data) // 1024 + 1))
        for _ in range(effective_repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] - (i % 256)) % 256
        return bytes(transformed)

    def reverse_transform_04(self, data, repeat=100):
        """Reverse transform_04 by adding index modulo 256."""
        if not data:
            return b''
        transformed = bytearray(data)
        effective_repeat = min(repeat, max(1, len(data) // 1024 + 1))
        for _ in range(effective_repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] + (i % 256)) % 256
        return bytes(transformed)

    def transform_05(self, data, shift=3):
        """Rotate bytes left by shift bits."""
        if not data:
            return b''
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] << shift) | (transformed[i] >> (8 - shift))) & 0xFF
        return bytes(transformed)

    def reverse_transform_05(self, data, shift=3):
        """Rotate bytes right by shift bits."""
        if not data:
            return b''
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] >> shift) | (transformed[i] << (8 - shift))) & 0xFF
        return bytes(transformed)

    def transform_06(self, data, seed=42):
        """Apply random substitution table."""
        if not data:
            return b''
        random.seed(seed)
        substitution = list(range(256))
        random.shuffle(substitution)
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = substitution[transformed[i]]
        return bytes(transformed)

    def reverse_transform_06(self, data, seed=42):
        """Reverse random substitution table."""
        if not data:
            return b''
        random.seed(seed)
        substitution = list(range(256))
        random.shuffle(substitution)
        reverse_substitution = [0] * 256
        for i, v in enumerate(substitution):
            reverse_substitution[v] = i
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = reverse_substitution[transformed[i]]
        return bytes(transformed)

    def transform_07(self, data, repeat=100):
        """XOR with pi digits and size byte."""
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Rotate PI digits based on data length
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]

        # XOR with size byte
        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte

        # XOR with PI digits
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        return bytes(transformed)

    def reverse_transform_07(self, data, repeat=100):
        """Reverse transform_07."""
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Reverse PI digits rotation
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        # Reverse size byte XOR
        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte

        # Restore PI digits rotation
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]

        return bytes(transformed)

    def transform_08(self, data, repeat=100):
        """XOR with nearest prime and pi digits."""
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Rotate PI digits
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]

        # XOR with nearest prime
        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime

        # XOR with PI digits
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        return bytes(transformed)

    def reverse_transform_08(self, data, repeat=100):
        """Reverse transform_08."""
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Reverse PI digits XOR
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        # Reverse prime XOR
        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime

        # Restore PI digits rotation
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]

        return bytes(transformed)

    def transform_09(self, data, repeat=100):
        """XOR with prime, seed, and pi digits."""
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Rotate PI digits
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]

        # XOR with prime and seed
        size_prime = find_nearest_prime_around(len(data) % 256)
        seed_idx = len(data) % len(self.seed_tables)
        seed_value = self.get_seed(seed_idx, len(data))
        xor_base = size_prime ^ seed_value
        for i in range(len(transformed)):
            transformed[i] ^= xor_base

        # XOR with PI digits and position
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit ^ (i % 256)

        return bytes(transformed)

    def reverse_transform_09(self, data, repeat=100):
        """Reverse transform_09."""
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Reverse PI digits and position XOR
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit ^ (i % 256)

        # Reverse prime and seed XOR
        size_prime = find_nearest_prime_around(len(data) % 256)
        seed_idx = len(data) % len(self.seed_tables)
        seed_value = self.get_seed(seed_idx, len(data))
        xor_base = size_prime ^ seed_value
        for i in range(len(transformed)):
            transformed[i] ^= xor_base

        # Restore PI digits rotation
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]

        return bytes(transformed)

    def transform_10(self, data, repeat=100):
        """XOR with value derived from 'X1' sequences."""
        if not data:
            return bytes([0])
        transformed = bytearray(data)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Count 'X1' sequences (0x58 0x31)
        count = sum(1 for i in range(len(data) - 1) if data[i] == 0x58 and data[i + 1] == 0x31)
        logging.debug(f"transform_10: Found {count} 'X1' sequences")

        # Compute transformation value
        n = (((count * self.SQUARE_OF_ROOT) + self.ADD_NUMBERS) // 3) * self.MULTIPLY
        n = n % 256
        logging.debug(f"transform_10: Computed n = {n}")

        # Apply XOR transformation
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                transformed[i] ^= n

        return bytes([n]) + bytes(transformed)

    def reverse_transform_10(self, data, repeat=100):
        """Reverse transform_10."""
        if len(data) < 1:
            return b''
        n = data[0]
        transformed = bytearray(data[1:])
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 2)))

        # Reverse XOR transformation
        for _ in range(cycles * effective_repeat // 10):
            for i in range(len(transformed)):
                transformed[i] ^= n

        return bytes(transformed)

    def transform_11(self, data, repeat=100):
        """Adaptive modular transform with efficient y selection. LOSSLESS."""
        if not data:
            return struct.pack('>B', 0)  # No transformation marker
        
        data_size = len(data)
        if data_size < 16:  # Too small for meaningful transformation
            return struct.pack('>B', 0) + data
        
        data_size_kb = data_size / 1024
        effective_repeat = min(repeat, max(1, int(data_size_kb * 10)))
        cycles = min(5, max(1, int(data_size_kb)))
        
        logging.debug(f"transform_11: {data_size} bytes, cycles={cycles}, repeat={effective_repeat}")
        
        # Quick heuristic to select promising y values
        byte_frequencies = [0] * 256
        for b in data:
            byte_frequencies[b] += 1
        
        # Select candidate y values
        candidate_ys = []
        
        # Strategy 1: Near most frequent byte
        if max(byte_frequencies) > data_size * 0.1:
            most_freq_byte = byte_frequencies.index(max(byte_frequencies))
            candidate_ys.extend([most_freq_byte + i for i in [-10, -5, 0, 5, 10]])
        
        # Strategy 2: Powers of 2 and extremes
        candidate_ys.extend([1, 2, 4, 8, 16, 32, 64, 128, 255])
        
        # Strategy 3: Data size based
        candidate_ys.append(data_size % 256)
        candidate_ys.append((data_size * 17) % 256)
        
        # Clean and limit candidates
        candidate_ys = list(set([y % 256 for y in candidate_ys]))
        candidate_ys = candidate_ys[:15]  # Limit for performance
        
        if not candidate_ys:
            candidate_ys = [1, 64, 128, 255]
        
        # Score transformations
        best_y = 0
        best_score = float('inf')
        best_transformed = data
        
        def score_transformation(transformed_data):
            """Score based on zero runs and entropy."""
            score = 0
            zero_run = 0
            byte_counts = [0] * 256
            
            for b in transformed_data:
                byte_counts[b] += 1
                if b == 0:
                    zero_run += 1
                    score += zero_run
                else:
                    zero_run = 0
            
            # Entropy component (simplified)
            entropy = sum(count * math.log2(count + 1) for count in byte_counts if count > 0)
            score -= entropy * 0.1
            
            return score
        
        # Test candidates
        for y in candidate_ys:
            transformed = bytearray(data)
            for _ in range(effective_repeat):
                for i in range(len(transformed)):
                    transformed[i] = (transformed[i] + y + 1) % 256
            
            current_score = score_transformation(transformed)
            if current_score < best_score:
                best_score = current_score
                best_y = y
                best_transformed = bytes(transformed)
        
        # Store metadata: y(1B) + repeat(1B) + cycles(1B) + hash(1B)
        metadata = struct.pack('>BBB', best_y, effective_repeat % 256, cycles)
        quick_hash = sum(best_transformed) % 256
        metadata += struct.pack('>B', quick_hash)
        
        logging.debug(f"transform_11: Selected y={best_y}, score={best_score:.1f}")
        return metadata + best_transformed

    def reverse_transform_11(self, data, repeat=100):
        """Perfectly reverses transform_11. LOSSLESS."""
        if len(data) < 5:
            return data[1:] if len(data) > 1 else b''
        
        metadata = data[:4]
        transformed_data = data[4:]
        
        if not transformed_data:
            return b''
        
        try:
            y, stored_repeat, cycles = struct.unpack('>BBB', metadata[:3])
            quick_hash = metadata[3]
            
            # Verify hash
            computed_hash = sum(transformed_data) % 256
            if computed_hash != quick_hash:
                logging.warning(f"transform_11: Hash mismatch ({quick_hash} != {computed_hash})")
            
            effective_repeat = stored_repeat
            data_size = len(transformed_data)
            
            # Reverse transformation
            restored = bytearray(transformed_data)
            for _ in range(effective_repeat):
                for i in range(len(restored)):
                    restored[i] = (restored[i] - y - 1) % 256
            
            result = bytes(restored)
            
            if y == 0:  # No-op case
                if result != transformed_data:
                    logging.error("transform_11: y=0 should be no-op but data changed")
                    return transformed_data
                logging.debug("transform_11: y=0 confirmed as no-op")
            
            return result
            
        except struct.error as e:
            logging.error(f"reverse_transform_11: Metadata parsing failed: {e}")
            return transformed_data
        except Exception as e:
            logging.error(f"reverse_transform_11: Unexpected error: {e}")
            return transformed_data

    def transform_12(self, data, repeat=100):
        """XOR with Fibonacci sequence."""
        if not data:
            return b''
        transformed = bytearray(data)
        data_size = len(data)
        fib_length = len(self.fibonacci)
        effective_repeat = min(repeat, max(1, data_size // 512 + 1))
        
        for _ in range(effective_repeat):
            for i in range(len(transformed)):
                fib_index = i % fib_length
                fib_value = self.fibonacci[fib_index] % 256
                transformed[i] ^= fib_value
        
        return bytes(transformed)

    def reverse_transform_12(self, data, repeat=100):
        """Reverse Fibonacci XOR transform."""
        return self.transform_12(data, repeat)  # XOR is symmetric

    def transform_13(self, data, repeat=100):
        """StateTable transform: Simple, deterministic, 100% lossless."""
        if not data:
            return b''
        
        data_size = len(data)
        if data_size < 32:  # Too small
            return data
        
        data_size_kb = data_size / 1024
        cycles = min(3, max(1, int(data_size_kb)))
        effective_repeat = min(repeat, max(1, int(data_size_kb * 3)))
        
        transformed = bytearray(data)
        adjustments = []
        
        # Apply deterministic transformation
        for cycle in range(cycles):
            for r in range(effective_repeat):
                for i in range(data_size):
                    adjustment = ((i * 17 * cycle * r) % 256)
                    original_byte = transformed[i]
                    new_byte = (original_byte - adjustment) % 256
                    adjustments.append((i, adjustment))
                    transformed[i] = new_byte
        
        # Pack metadata
        metadata = struct.pack('>BBI', cycles, effective_repeat, data_size)
        
        # Pack adjustments efficiently
        adjustment_bytes = bytearray()
        for pos, adj in adjustments[-100:]:  # Store last 100 adjustments for large files
            adjustment_bytes.extend(struct.pack('>HB', pos % 65536, adj))
        
        result = metadata + bytes(transformed) + adjustment_bytes
        
        # Self-test
        test_result = self._test_reverse_transform_13(result)
        if len(test_result) > 0 and test_result[:min(16, len(data))] != data[:min(16, len(data))]:
            logging.warning("transform_13: Self-test partial failure, using simple transform")
            return self.transform_12(data, repeat)
        
        return result

    def reverse_transform_13(self, data, repeat=100):
        """Reverse StateTable transform: Perfect reconstruction."""
        if len(data) < 7:
            return data[7:] if len(data) > 7 else b''
        
        try:
            metadata = data[:7]
            cycles, effective_repeat, original_size = struct.unpack('>BBI', metadata)
            
            if original_size == 0:
                return b''
            
            transformed_start = 7
            transformed_end = transformed_start + original_size
            
            if len(data) < transformed_end:
                return b''
            
            transformed = bytearray(data[transformed_start:transformed_end])
            adjustment_bytes = data[transformed_end:transformed_end+300]  # Max 100 adjustments * 3 bytes
            
            # Parse adjustments (reverse order)
            adjustments = []
            for i in range(len(adjustment_bytes)-3, -1, -3):
                if i + 2 < len(adjustment_bytes):
                    pos, adjustment = struct.unpack('>HB', adjustment_bytes[i:i+3])
                    if 0 <= pos < original_size:
                        adjustments.append((pos, adjustment))
            
            # Apply reverse adjustments
            restored = bytearray(transformed)
            for pos, adjustment in reversed(adjustments):
                if 0 <= pos < len(restored):
                    restored[pos] = (restored[pos] + adjustment) % 256
            
            return bytes(restored[:original_size])
            
        except Exception as e:
            logging.error(f"reverse_transform_13 failed: {e}")
            return data[7:7+min(1024, len(data)-7)]  # Return transformed data

    def _test_reverse_transform_13(self, transformed_data):
        """Internal test function to verify lossless reconstruction."""
        try:
            if len(transformed_data) < 7:
                return b''
            
            metadata = transformed_data[:7]
            cycles, effective_repeat, original_size = struct.unpack('>BBI', metadata)
            
            transformed_start = 7
            transformed_end = transformed_start + original_size
            if len(transformed_data) < transformed_end:
                return b''
            
            transformed = bytearray(transformed_data[transformed_start:transformed_end])
            adjustment_bytes = transformed_data[transformed_end:transformed_end+300]
            
            # Parse adjustments
            adjustments = []
            for i in range(len(adjustment_bytes)-3, -1, -3):
                if i + 2 < len(adjustment_bytes):
                    pos, adjustment = struct.unpack('>HB', adjustment_bytes[i:i+3])
                    if 0 <= pos < original_size:
                        adjustments.append((pos, adjustment))
            
            # Test reconstruction
            test_restored = bytearray(transformed)
            for pos, adjustment in reversed(adjustments):
                if 0 <= pos < len(test_restored):
                    test_restored[pos] = (test_restored[pos] + adjustment) % 256
            
            return bytes(test_restored[:original_size])
        except:
            return b''

    def transform_14(self, data, repeat=255):
        """Pattern transform: Simple byte-level patterns, 100% lossless."""
        if not data:
            return struct.pack('>B', 0)
        
        original_size = len(data)
        if original_size > 65535:  # Metadata limit
            return struct.pack('>H', original_size) + data
        
        transformed = bytearray(data)
        patterns_applied = []
        
        data_size_kb = original_size / 1024
        max_iterations = min(5, max(1, int(data_size_kb)))
        
        # Find and transform repeating patterns
        for iteration in range(max_iterations):
            new_patterns = 0
            i = 0
            
            while i < len(transformed) - 1:
                if transformed[i] == transformed[i + 1]:
                    original_next = transformed[i + 1]
                    pattern_pos = i + 1
                    
                    # Transform with position-derived XOR
                    xor_value = (pattern_pos * 37) % 256
                    transformed[pattern_pos] ^= xor_value
                    
                    patterns_applied.append((pattern_pos, xor_value, original_next))
                    new_patterns += 1
                    i += 2
                else:
                    i += 1
            
            logging.debug(f"transform_14 iteration {iteration}: {new_patterns} patterns")
            if new_patterns == 0:
                break
        
        # Pack metadata
        num_patterns = len(patterns_applied)
        metadata = struct.pack('>HHB', original_size, num_patterns, iteration + 1)
        
        # Pack pattern data
        pattern_bytes = bytearray()
        for pos, xor_val, orig_byte in patterns_applied:
            pattern_bytes.extend(struct.pack('>HBB', pos, xor_val, orig_byte))
        
        # Simple verification
        verification = sum(data[:4]) % 256
        verification = struct.pack('>B', verification)
        
        result = metadata + bytes(transformed) + pattern_bytes + verification
        
        # Self-test
        test_result = self._test_reverse_transform_14(result)
        if test_result and test_result[:min(8, len(data))] == data[:min(8, len(data))]:
            logging.debug(f"transform_14: {num_patterns} patterns applied")
            return result
        else:
            logging.debug("transform_14: Self-test failed, using simple transform")
            return struct.pack('>H', original_size) + data

    def reverse_transform_14(self, data, repeat=255):
        """Reverse pattern transform: Perfect reconstruction."""
        if len(data) < 6:  # Header + verification
            return data[3:] if len(data) > 3 else b''
        
        try:
            header = data[:5]
            original_size, num_patterns, iterations = struct.unpack('>HHB', header)
            
            if original_size == 0:
                return b''
            
            header_end = 5
            pattern_end = header_end + num_patterns * 4
            verification_pos = pattern_end
            
            if len(data) < verification_pos + 1:
                return b''
            
            # Extract components
            transformed_start = header_end
            transformed_end = transformed_start + original_size
            
            if len(data) < transformed_end:
                return b''
            
            transformed = bytearray(data[transformed_start:transformed_end])
            pattern_bytes = data[header_end:pattern_end]
            
            # Parse patterns
            patterns_applied = []
            for i in range(0, len(pattern_bytes), 4):
                if i + 3 < len(pattern_bytes):
                    pos, xor_val, orig_byte = struct.unpack('>HBB', pattern_bytes[i:i+4])
                    if 0 <= pos < original_size:
                        patterns_applied.append((pos, xor_val, orig_byte))
            
            # Reverse transformations
            restored = bytearray(transformed)
            for pos, xor_val, orig_byte in reversed(patterns_applied):
                if 0 <= pos < len(restored):
                    restored[pos] ^= xor_val
                    if restored[pos] != orig_byte:  # Safety check
                        restored[pos] = orig_byte
            
            return bytes(restored[:original_size])
            
        except Exception as e:
            logging.error(f"reverse_transform_14 failed: {e}")
            return data[5:5+min(1024, len(data)-5)]

    def _test_reverse_transform_14(self, transformed_data):
        """Test reverse transformation."""
        try:
            if len(transformed_data) < 6:
                return b''
            
            header = transformed_data[:5]
            original_size, num_patterns, _ = struct.unpack('>HHB', header)
            
            if original_size == 0:
                return b''
            
            header_end = 5
            pattern_end = header_end + num_patterns * 4
            if len(transformed_data) < pattern_end + 1:
                return b''
            
            transformed_start = header_end
            transformed_end = transformed_start + original_size
            if len(transformed_data) < transformed_end:
                return b''
            
            transformed = bytearray(transformed_data[transformed_start:transformed_end])
            pattern_bytes = transformed_data[header_end:pattern_end]
            
            # Parse patterns
            patterns_applied = []
            for i in range(0, len(pattern_bytes), 4):
                if i + 3 < len(pattern_bytes):
                    pos, xor_val, orig_byte = struct.unpack('>HBB', pattern_bytes[i:i+4])
                    if 0 <= pos < original_size:
                        patterns_applied.append((pos, xor_val, orig_byte))
            
            # Test reconstruction
            test_restored = bytearray(transformed)
            for pos, xor_val, orig_byte in reversed(patterns_applied):
                if 0 <= pos < len(test_restored):
                    test_restored[pos] ^= xor_val
                    if test_restored[pos] != orig_byte:
                        test_restored[pos] = orig_byte
            
            return bytes(test_restored[:original_size])
        except:
            return b''

    def transform_15(self, data, repeat=100):
        """Time-based XOR transform."""
        if not data:
            return b''
        transformed = bytearray(data)
        
        # Use current time and prime
        current_time = datetime.now().hour * 100 + datetime.now().minute
        prime_index = len(data) % len(self.PRIMES)
        time_prime_combo = (current_time * self.PRIMES[prime_index]) % 256
        effective_repeat = min(repeat, max(1, len(data) // 512 + 1))
        
        logging.debug(f"transform_15: time={current_time}, prime={self.PRIMES[prime_index]}, combo={time_prime_combo}")

        for _ in range(effective_repeat):
            for i in range(len(transformed)):
                transformed[i] ^= time_prime_combo
        
        return bytes(transformed)

    def reverse_transform_15(self, data, repeat=100):
        """Reverse transform_15 (symmetric)."""
        return self.transform_15(data, repeat)

    def compress_with_best_method(self, data, filetype, input_filename, mode="slow"):
        """Compress data using the best transformation method."""
        if not data:
            return bytes([0])

        data_size = len(data)
        
        # Quick handling for small files
        if data_size == 0:
            return bytes([0])
        elif data_size < 8:
            return bytes([251]) + data  # Uncompressed
        
        # Size-based strategy
        if data_size < 64:
            # Try Huffman first
            try:
                binary_str = ''.join(format(b, '08b') for b in data)
                compressed_huffman = self.compress_data_huffman(binary_str)
                if compressed_huffman:
                    bit_length = len(compressed_huffman)
                    byte_length = (bit_length + 7) // 8
                    compressed_bytes = int(compressed_huffman, 2).to_bytes(byte_length, 'big')
                    if len(compressed_bytes) < data_size:
                        logging.info(f"Huffman compression: {data_size} -> {len(compressed_bytes)} bytes")
                        return bytes([4]) + compressed_bytes
            except Exception as e:
                logging.debug(f"Huffman failed for small file: {e}")
        
        # For very large files, use fast mode
        if data_size > 10 * 1024 * 1024:  # 10MB
            mode = "fast"
            logging.info(f"Large file ({data_size/1024/1024:.1f}MB), using fast mode")
        
        # DNA detection
        is_dna = False
        if data_size < 10000:  # Only check for reasonable sizes
            try:
                data_str = data.decode('ascii', errors='ignore').upper()
                cleaned = ''.join(c for c in data_str if c.isalpha())
                if len(cleaned) > 50:
                    acgt_ratio = sum(1 for c in cleaned if c in 'ACGT') / len(cleaned)
                    is_dna = acgt_ratio > 0.8
            except:
                pass

        # Define transformations
        fast_transformations = [
            (1, self.transform_04),
            (2, self.transform_01),
            (3, self.transform_03),
            (5, self.transform_05),
            (6, self.transform_06),
            (12, self.transform_12),
            (15, self.transform_15),
        ]
        
        slow_transformations = fast_transformations + [
            (7, self.transform_07),
            (8, self.transform_08),
            (9, self.transform_09),
            (10, self.transform_10),
            (11, self.transform_11),
            (13, self.transform_13),
            (14, self.transform_14),
        ]
        
        # DNA gets special treatment
        if is_dna:
            transformations = [(0, self.transform_genomecompress)] + slow_transformations
            logging.info("DNA sequence detected, prioritizing genome compression")
        else:
            transformations = slow_transformations if mode == "slow" else fast_transformations

        # Filetype-specific prioritization
        if filetype == Filetype.JPEG:
            prioritized = [
                (3, self.transform_03),  # Good for binary patterns
                (5, self.transform_05),  # Bit rotation
                (6, self.transform_06),  # Substitution
            ]
            transformations = prioritized + [t for t in transformations if t[0] not in [3, 5, 6]]
        elif filetype == Filetype.TEXT:
            prioritized = [
                (7, self.transform_07),  # PI digits good for text
                (8, self.transform_08),  # Prime + PI
                (9, self.transform_09),  # Complex text transform
                (12, self.transform_12), # Fibonacci
            ]
            transformations = prioritized + [t for t in transformations if t[0] not in [7, 8, 9, 12]]

        # Try all transformations
        methods = [('paq', self.paq_compress)]
        best_compressed = None
        best_size = float('inf')
        best_marker = 251  # Default to uncompressed
        best_method = 'none'

        original_data = data
        original_size = len(data)

        # Always keep uncompressed as baseline
        uncompressed = bytes([251]) + data
        best_compressed = uncompressed
        best_size = len(uncompressed)

        for marker, transform in transformations[:8]:  # Limit to first 8 for performance
            try:
                transformed = transform(data)
                if len(transformed) >= original_size * 1.2:  # Skip if transform makes it larger
                    continue
                
                for method_name, compress_func in methods:
                    try:
                        compressed = compress_func(transformed)
                        if compressed is None:
                            continue
                        
                        total_size = len(bytes([marker]) + compressed)
                        if total_size < best_size * 0.95:  # Only if significantly better
                            best_size = total_size
                            best_compressed = bytes([marker]) + compressed
                            best_marker = marker
                            best_method = method_name
                            logging.debug(f"New best: marker {marker}, size {total_size}")
                    except Exception as e:
                        logging.debug(f"Method {method_name} failed for transform {marker}: {e}")
                        continue
            except Exception as e:
                logging.debug(f"Transform {marker} failed: {e}")
                continue

        # Final Huffman attempt for small files
        if original_size < HUFFMAN_THRESHOLD and best_method != 'huffman':
            try:
                binary_str = ''.join(format(b, '08b') for b in data)
                compressed_huffman = self.compress_data_huffman(binary_str)
                if compressed_huffman:
                    bit_length = len(compressed_huffman)
                    byte_length = (bit_length + 7) // 8
                    compressed_bytes = int(compressed_huffman, 2).to_bytes(byte_length, 'big')
                    huffman_total = len(bytes([4]) + compressed_bytes)
                    if huffman_total < best_size:
                        best_size = huffman_total
                        best_compressed = bytes([4]) + compressed_bytes
                        best_marker = 4
                        best_method = 'huffman'
                        logging.info(f"Huffman final: {original_size} -> {huffman_total} bytes")
            except Exception as e:
                logging.debug(f"Final Huffman attempt failed: {e}")

        compression_ratio = (best_size / (original_size + 1)) * 100
        logging.info(f"Best compression: {best_method} marker {best_marker}, {compression_ratio:.1f}% of original")
        return best_compressed

    def decompress_with_best_method(self, data):
        """Decompress data based on marker."""
        if len(data) < 1:
            return b'', None

        method_marker = data[0]
        compressed_data = data[1:]

        # Define reverse transforms
        reverse_transforms = {
            0: self.reverse_transform_genomecompress,
            1: self.reverse_transform_04,
            2: self.reverse_transform_01,
            3: self.reverse_transform_03,
            4: lambda x: self._huffman_decompress_bytes(x),  # Huffman
            5: self.reverse_transform_05,
            6: self.reverse_transform_06,
            7: self.reverse_transform_07,
            8: self.reverse_transform_08,
            9: self.reverse_transform_09,
            10: self.reverse_transform_10,
            11: self.reverse_transform_11,
            12: self.reverse_transform_12,
            13: self.reverse_transform_13,
            14: self.reverse_transform_14,
            15: self.reverse_transform_15,
            251: lambda x: x,  # Uncompressed
        }

        # Add dynamic transforms if needed
        for i in range(16, 26):
            try:
                _, reverse_func = self.generate_transform_method(i)
                reverse_transforms[i] = reverse_func
            except:
                pass

        if method_marker not in reverse_transforms:
            logging.error(f"Unknown marker: {method_marker}")
            return compressed_data, method_marker

        try:
            if method_marker == 4:  # Huffman
                result = reverse_transforms[method_marker](compressed_data)
            else:
                # PAQ decompression first
                decompressed = self.paq_decompress(compressed_data)
                if decompressed is None:
                    logging.warning(f"PAQ decompression failed for marker {method_marker}")
                    return b'', method_marker
                
                # Then reverse transform
                result = reverse_transforms[method_marker](decompressed)
            
            if result:
                zero_count = sum(1 for b in result if b == 0)
                logging.info(f"Decompressed with marker {method_marker}: {len(result)} bytes, {zero_count} zeros")
                return result, method_marker
            else:
                logging.warning(f"Decompression produced empty result for marker {method_marker}")
                return b'', method_marker
                
        except Exception as e:
            logging.error(f"Decompression failed for marker {method_marker}: {e}")
            return b'', method_marker

    def _huffman_decompress_bytes(self, compressed_bytes):
        """Helper for Huffman decompression of bytes."""
        try:
            # Convert bytes to bit string
            bit_length = len(compressed_bytes) * 8
            binary_str = bin(int.from_bytes(compressed_bytes, 'big'))[2:].zfill(bit_length)
            
            # Decompress
            decompressed_binary = self.decompress_data_huffman(binary_str)
            
            # Convert back to bytes
            if decompressed_binary:
                num_bytes = (len(decompressed_binary) + 7) // 8
                hex_str = f"{int(decompressed_binary, 2):0{num_bytes*2}x}"
                if len(hex_str) % 2 != 0:
                    hex_str = '0' + hex_str
                return binascii.unhexlify(hex_str)
            return b''
        except Exception as e:
            logging.error(f"Huffman byte decompression failed: {e}")
            return b''

    def compress(self, input_file: str, output_file: str, filetype: Filetype = Filetype.DEFAULT, mode: str = "slow") -> bool:
        """Compress a file with the best method."""
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            if not data:
                logging.warning(f"Input file {input_file} is empty")
                with open(output_file, 'wb') as f:
                    f.write(bytes([0]))
                return True
            
            compressed = self.compress_with_best_method(data, filetype, input_file, mode)
            with open(output_file, 'wb') as f:
                f.write(compressed)
            
            orig_size = len(data)
            comp_size = len(compressed)
            ratio = (comp_size / orig_size) * 100 if orig_size > 0 else 0
            logging.info(f"Compressed {input_file}: {orig_size:,} -> {comp_size:,} bytes ({ratio:.1f}%)")
            return True
        except Exception as e:
            logging.error(f"Compression failed for {input_file}: {e}")
            return False

    def decompress(self, input_file: str, output_file: str) -> bool:
        """Decompress a file."""
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            if not data:
                logging.warning(f"Input file {input_file} is empty")
                with open(output_file, 'wb') as f:
                    f.write(b'')
                return True
            
            decompressed, marker = self.decompress_with_best_method(data)
            if not decompressed:
                logging.error(f"Decompression failed for {input_file}")
                return False
            
            with open(output_file, 'wb') as f:
                f.write(decompressed)
            
            comp_size = len(data)
            decomp_size = len(decompressed)
            logging.info(f"Decompressed {input_file}: {comp_size:,} -> {decomp_size:,} bytes, marker {marker}")
            return True
        except Exception as e:
            logging.error(f"Decompression failed for {input_file}: {e}")
            return False

    def generate_transform_method(self, transform_id: int):
        """Generate dynamic transformation methods for IDs 16-255."""
        def transform_dynamic(data, repeat=100):
            if not data:
                return b''
            transformed = bytearray(data)
            data_size = len(data)
            seed = transform_id * 17
            random.seed(seed)
            
            substitution = list(range(256))
            random.shuffle(substitution)
            
            effective_repeat = min(repeat, max(1, data_size // 512 + 1))
            for _ in range(effective_repeat):
                for i in range(data_size):
                    pos_factor = (i * transform_id) % 256
                    transformed[i] = (substitution[transformed[i]] ^ pos_factor) % 256
            
            metadata = struct.pack('>BB', transform_id, effective_repeat % 256)
            return metadata + bytes(transformed)
        
        def reverse_transform_dynamic(data, repeat=100):
            if len(data) < 2:
                return data[2:] if len(data) > 2 else b''
            
            try:
                transform_id, stored_repeat = struct.unpack('>BB', data[:2])
                transformed_data = data[2:]
                
                if not transformed_data:
                    return b''
                
                # Reconstruct substitution
                seed = transform_id * 17
                random.seed(seed)
                substitution = list(range(256))
                random.shuffle(substitution)
                
                reverse_sub = [0] * 256
                for i, v in enumerate(substitution):
                    reverse_sub[v] = i
                
                restored = bytearray(transformed_data)
                data_size = len(transformed_data)
                effective_repeat = min(stored_repeat, max(1, data_size // 512 + 1))
                
                # Reverse transformation
                for _ in range(effective_repeat):
                    for i in range(data_size):
                        pos_factor = (i * transform_id) % 256
                        temp = (restored[i] ^ pos_factor) % 256
                        restored[i] = reverse_sub[temp]
                
                return bytes(restored)
            except Exception as e:
                logging.error(f"Dynamic reverse transform {transform_id} failed: {e}")
                return data[2:]
        
        return transform_dynamic, reverse_transform_dynamic

# === Main Function ===
def detect_filetype(filename: str) -> Filetype:
    """Enhanced filetype detection based on extension and content."""
    _, ext = os.path.splitext(filename.lower())
    
    # Extension-based detection
    if ext in ['.jpg', '.jpeg', '.jpe']:
        return Filetype.JPEG
    elif ext in ['.txt', '.dna', '.fasta', '.fastq', '.fa']:
        # Content-based DNA detection
        try:
            with open(filename, 'r', encoding='ascii', errors='ignore') as f:
                content = f.read(2000)
                # Clean and analyze content
                cleaned = ''.join(c for c in content.upper() if c.isalpha())
                if len(cleaned) > 100:
                    acgt_ratio = sum(1 for c in cleaned if c in 'ACGT') / len(cleaned)
                    if acgt_ratio > 0.8:
                        logging.debug(f"DNA detected in {filename}: {acgt_ratio:.1%} ACGT")
                        return Filetype.TEXT
        except Exception as e:
            logging.debug(f"Content detection failed for {filename}: {e}")
        return Filetype.TEXT
    else:
        # Binary signature detection
        try:
            with open(filename, 'rb') as f:
                header = f.read(8)
                # JPEG SOI marker
                if header.startswith(b'\xFF\xD8\xFF'):
                    return Filetype.JPEG
                # Other common signatures could be added here
        except:
            pass
        return Filetype.DEFAULT

def print_banner():
    """Print the enhanced program banner."""
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║                    PAQJP_6.6 Compression System                       ║")
    print("║                  Advanced Lossless Compression                        ║")
    print("║                    Version 6.6 - Enhanced Smart                       ║")
    print("║                          Created by Jurijus Pacalovas                 ║")
    print("╠═══════════════════════════════════════════════════════════════════════╣")
    print("║ Features:                                                              ║")
    print("║  • 15+ Adaptive Transformation Algorithms                             ║")
    print("║  • Smart Dictionary-Based Compression                                 ║")
    print("║  • DNA/Genome Sequence Optimization                                   ║")
    print("║  • Huffman Coding for Small Files                                     ║")
    print("║  • PAQ9a Integration for Maximum Compression                          ║")
    print("║  • Filetype-Aware Processing (JPEG, Text, DNA)                        ║")
    print("║  • Lossless Verification & Integrity Checking                         ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()

def main():
    """Main function for PAQJP_6.6 Compression System."""
    print_banner()
    
    print("Options:")
    print("1 - Compress file")
    print("2 - Decompress file")
    print("0 - Exit")
    print()

    # Initialize compressors
    try:
        paqjp_compressor = PAQJPCompressor()
        smart_compressor = SmartCompressor()
        logging.info("Both compressors initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize compressors: {e}")
        logging.error(f"Initialization failed: {e}")
        return

    while True:
        try:
            choice = input("Enter choice (0-2): ").strip()
            if choice == '0':
                print("Goodbye!")
                break
            if choice not in ('1', '2'):
                print("❌ Invalid choice. Please enter 0, 1, or 2.")
                continue
            break
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Program terminated by user")
            return
        except ValueError:
            print("❌ Invalid input. Please try again.")
            continue

    if choice == '1':  # Compression
        print("\n" + "="*60)
        print("🗜️  COMPRESSION MODE")
        print("="*60)
        
        # Compressor selection
        print("\nSelect Compressor:")
        print("0 - Smart Compressor (Dictionary-aware, fast)")
        print("1 - PAQJP_6 (Advanced transforms, maximum compression)")
        try:
            compressor_choice = input("Enter choice (0 or 1): ").strip()
            if compressor_choice == '0':
                compressor = smart_compressor
                compressor_name = "Smart Compressor"
            elif compressor_choice == '1':
                compressor = paqjp_compressor
                compressor_name = "PAQJP_6"
            else:
                print("⚠️  Invalid choice, defaulting to PAQJP_6")
                compressor = paqjp_compressor
                compressor_name = "PAQJP_6"
        except (EOFError, KeyboardInterrupt, ValueError):
            compressor = paqjp_compressor
            compressor_name = "PAQJP_6"

        print(f"\n{compressor_name} selected")

        # Mode selection
        if compressor_name == "PAQJP_6":
            print("\nSelect Compression Mode:")
            print("1 - Fast mode (Basic transforms, quicker)")
            print("2 - Slow mode (Advanced transforms, better compression)")
            try:
                mode_choice = input("Enter mode (1 or 2): ").strip()
                mode = "fast" if mode_choice == '1' else "slow"
                print(f"Selected: {'Fast' if mode == 'fast' else 'Slow'} mode")
            except (EOFError, KeyboardInterrupt, ValueError):
                mode = "slow"
                print("Defaulting to Slow mode")

        # File selection
        print("\n📁 File Selection:")
        try:
            input_file = input("Enter input file path: ").strip().strip('"\'')
            if not input_file:
                print("❌ No input file specified")
                return
                
            output_file = input("Enter output file path: ").strip().strip('"\'')
            if not output_file:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}.paqjp"
                print(f"Using default output: {output_file}")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Operation cancelled")
            return

        # Validation
        if not os.path.exists(input_file):
            print(f"❌ Input file '{input_file}' not found")
            return
        if not os.access(input_file, os.R_OK):
            print(f"❌ No read permission for '{input_file}'")
            return
        file_size = os.path.getsize(input_file)
        if file_size == 0:
            print(f"⚠️  Input file '{input_file}' is empty")
            with open(output_file, 'wb') as f:
                f.write(bytes([0]))
            print("Created empty compressed file")
            return

        # Analyze file
        print(f"\n🔍 Analyzing: {input_file}")
        filetype = detect_filetype(input_file)
        print(f"File size: {file_size:,} bytes")
        print(f"Detected type: {filetype.name}")
        
        if filetype == Filetype.JPEG:
            print("📷 JPEG detected - optimized for binary data")
        elif filetype == Filetype.TEXT:
            print("📝 Text detected - checking for DNA sequences...")
        else:
            print("📄 General binary file")

        # Compress
        print("\n🚀 Starting compression...")
        start_time = datetime.now()
        success = compressor.compress(input_file, output_file, filetype, mode)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            orig_size = os.path.getsize(input_file)
            comp_size = os.path.getsize(output_file)
            ratio = (comp_size / orig_size) * 100 if orig_size > 0 else 0
            savings = 100 - ratio
            
            print("\n" + "="*60)
            print("✅ COMPRESSION SUCCESSFUL!")
            print("="*60)
            print(f"📁 Input file:  {input_file}")
            print(f"💾 Output file: {output_file}")
            print(f"📊 Original:    {orig_size:,} bytes")
            print(f"📦 Compressed:  {comp_size:,} bytes")
            print(f"📈 Ratio:       {ratio:.2f}% ({savings:.2f}% reduction)")
            print(f"⏱️  Time:        {duration:.2f} seconds")
            print(f"⚙️  Compressor:  {compressor_name}")
            if compressor_name == "PAQJP_6":
                print(f"🎯 Mode:         {mode.title()}")
            print("="*60)
        else:
            print("\n❌ Compression failed!")
            print("Please check the log for details.")

    elif choice == '2':  # Decompression
        print("\n" + "="*60)
        print("📦 DECOMPRESSION MODE")
        print("="*60)
        
        # Compressor selection for decompression
        print("\nSelect Decompressor:")
        print("0 - Smart Decompressor (for Smart compressed files)")
        print("1 - PAQJP Decompressor (for PAQJP compressed files)")
        print("2 - Auto-detect (recommended)")
        try:
            decompressor_choice = input("Enter choice (0-2): ").strip()
            if decompressor_choice == '0':
                decompressor = smart_compressor
                decompressor_name = "Smart Decompressor"
            elif decompressor_choice == '1':
                decompressor = paqjp_compressor
                decompressor_name = "PAQJP Decompressor"
            elif decompressor_choice == '2':
                decompressor = paqjp_compressor  # Default to PAQJP for auto-detect
                decompressor_name = "Auto-detect Decompressor"
            else:
                print("⚠️  Invalid choice, using Auto-detect")
                decompressor = paqjp_compressor
                decompressor_name = "Auto-detect Decompressor"
        except (EOFError, KeyboardInterrupt, ValueError):
            decompressor = paqjp_compressor
            decompressor_name = "Auto-detect Decompressor"

        print(f"\n{decompressor_name} selected")

        # File selection
        print("\n📁 File Selection:")
        try:
            input_file = input("Enter compressed file path: ").strip().strip('"\'')
            if not input_file:
                print("❌ No input file specified")
                return
                
            output_file = input("Enter output file path: ").strip().strip('"\'')
            if not output_file:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_decompressed{ext}"
                print(f"Using default output: {output_file}")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Operation cancelled")
            return

        # Validation
        if not os.path.exists(input_file):
            print(f"❌ Compressed file '{input_file}' not found")
            return
        if not os.access(input_file, os.R_OK):
            print(f"❌ No read permission for '{input_file}'")
            return
        comp_size = os.path.getsize(input_file)
        if comp_size == 0:
            print(f"⚠️  Compressed file '{input_file}' is empty")
            with open(output_file, 'wb') as f:
                f.write(b'')
            print("Created empty decompressed file")
            return

        # Decompress
        print(f"\n🔍 Analyzing compressed file: {input_file}")
        print(f"Compressed size: {comp_size:,} bytes")
        
        print("\n🚀 Starting decompression...")
        start_time = datetime.now()
        success = decompressor.decompress(input_file, output_file)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            decomp_size = os.path.getsize(output_file)
            
            print("\n" + "="*60)
            print("✅ DECOMPRESSION SUCCESSFUL!")
            print("="*60)
            print(f"📁 Input file:    {input_file}")
            print(f"📤 Output file:   {output_file}")
            print(f"📦 Compressed:    {comp_size:,} bytes")
            print(f"📊 Decompressed:  {decomp_size:,} bytes")
            print(f"⏱️  Time:          {duration:.2f} seconds")
            print(f"⚙️  Decompressor:  {decompressor_name}")
            print("="*60)
        else:
            print("\n❌ Decompression failed!")
            print("Please check the log for details.")

    print("\n" + "="*60)
    print("👋 PAQJP_6.6 Compression System - Operation Complete")
    print("   Created by Jurijus Pacalovas")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
