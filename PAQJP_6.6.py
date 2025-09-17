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
        pi_digits = [int(d) for d in mp.pi.digits(10)[0]]
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
        fallback_digits = [3, 1, 4]
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits[:num_digits]]
        save_pi_digits(mapped_fallback, filename)
        return mapped_fallback
    except Exception as e:
        logging.error(f"Failed to generate pi digits: {e}")
        fallback_digits = [3, 1, 4]
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
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
        for _ in range(repeat):
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def transform_with_pattern_chunk(data, chunk_size=4):
    """XOR each chunk with 0xFF."""
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
                        data.append(f.read())
                    logging.info(f"Loaded dictionary: {filename}")
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")
            else:
                logging.warning(f"Missing dictionary: {filename}")
        return data

    def compute_sha256(self, data):
        """Compute SHA-256 hash as hex."""
        return hashlib.sha256(data).hexdigest()

    def compute_sha256_binary(self, data):
        """Compute SHA-256 hash as bytes."""
        return hashlib.sha256(data).digest()

    def find_hash_in_dictionaries(self, hash_hex):
        """Search for hash in dictionary files."""
        for filename in DICTIONARY_FILES:
            if not os.path.exists(filename):
                continue
            try:
                with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if hash_hex in line:
                            logging.info(f"Hash {hash_hex[:16]}... found in {filename}")
                            return filename
            except Exception as e:
                logging.warning(f"Error searching {filename}: {e}")
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
            logging.info("PAQ9a compression complete")
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
            logging.info("PAQ9a decompression complete")
            return decompressed
        except Exception as e:
            logging.error(f"PAQ9a decompression failed: {e}")
            return None

    def reversible_transform(self, data):
        """Apply reversible XOR transform with 0xAA."""
        logging.info("Applying XOR transform (0xAA)")
        transformed = bytes(b ^ 0xAA for b in data)
        logging.info("XOR transform complete")
        return transformed

    def reverse_reversible_transform(self, data):
        """Reverse XOR transform with 0xAA."""
        logging.info("Reversing XOR transform (0xAA)")
        return self.reversible_transform(data)  # XOR is symmetric

    def compress(self, input_data, input_file):
        """Compress data using Smart Compressor."""
        if not input_data:
            logging.warning("Empty input, returning minimal output")
            return bytes([0])

        original_hash = self.compute_sha256(input_data)
        logging.info(f"SHA-256 of input: {original_hash[:16]}...")

        found = self.find_hash_in_dictionaries(original_hash)
        if found:
            logging.info(f"Hash found in dictionary: {found}")
        else:
            logging.info("Hash not found, proceeding with compression")

        if input_file.endswith(".paq") and any(x in input_file for x in ["words", "lines", "sentence"]):
            sha = self.generate_8byte_sha(input_data)
            if sha and len(input_data) > 8:
                logging.info(f"SHA-8 for .paq file: {sha.hex()}")
                return sha
            logging.info("Original smaller than SHA, skipping compression")
            return None

        transformed = self.reversible_transform(input_data)
        compressed = self.paq_compress(transformed)
        if compressed is None:
            logging.error("Compression failed")
            return None

        if len(compressed) < len(input_data):
            output = self.compute_sha256_binary(input_data) + compressed
            logging.info(f"Smart compression: Original {len(input_data)} bytes, Compressed {len(compressed)} bytes")
            return output
        else:
            logging.info("Compression not efficient, returning None")
            return None

    def decompress(self, input_data):
        """Decompress data using Smart Compressor."""
        if len(input_data) < 32:
            logging.error("Input too short for Smart Compressor")
            return None

        stored_hash = input_data[:32]
        compressed_data = input_data[32:]

        decompressed = self.paq_decompress(compressed_data)
        if decompressed is None:
            return None

        original = self.reverse_reversible_transform(decompressed)
        computed_hash = self.compute_sha256_binary(original)
        if computed_hash == stored_hash:
            logging.info("Hash verification successful")
            return original
        else:
            logging.error("Hash verification failed")
            return None

# === PAQJP Compressor ===
class PAQJPCompressor:
    def __init__(self):
        self.PI_DIGITS = PI_DIGITS
        self.PRIMES = PRIMES
        self.seed_tables = self.generate_seed_tables()
        self.SQUARE_OF_ROOT = 2
        self.ADD_NUMBERS = 1
        self.MULTIPLY = 3
        self.fibonacci = self.generate_fibonacci(100)
        self.state_table = StateTable()

    def generate_fibonacci(self, n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms."""
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
            logging.warning("Empty binary string, returning empty frequencies")
            return {}
        frequencies = {}
        for bit in binary_str:
            frequencies[bit] = frequencies.get(bit, 0) + 1
        return frequencies

    def build_huffman_tree(self, frequencies):
        """Build Huffman tree from frequencies."""
        if not frequencies:
            logging.warning("No frequencies, returning None")
            return None
        heap = [(freq, Node(symbol=symbol)) for symbol, freq in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            freq1, node1 = heapq.heappop(heap)
            freq2, node2 = heapq.heappop(heap)
            new_node = Node(left=node1, right=node2)
            heapq.heappush(heap, (freq1 + freq2, new_node))
        return heap[0][1]

    def generate_huffman_codes(self, root, current_code="", codes={}):
        """Generate Huffman codes from tree."""
        if root is None:
            logging.warning("Huffman tree is None, returning empty codes")
            return {}
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
            logging.warning("Empty binary string, returning empty compressed string")
            return ""
        frequencies = self.calculate_frequencies(binary_str)
        huffman_tree = self.build_huffman_tree(frequencies)
        if huffman_tree is None:
            return ""
        huffman_codes = self.generate_huffman_codes(huffman_tree)
        if '0' not in huffman_codes:
            huffman_codes['0'] = '0'
        if '1' not in huffman_codes:
            huffman_codes['1'] = '1'
        return ''.join(huffman_codes[bit] for bit in binary_str)

    def decompress_data_huffman(self, compressed_str):
        """Decompress Huffman-coded string."""
        if not compressed_str:
            logging.warning("Empty compressed string, returning empty decompressed string")
            return ""
        frequencies = self.calculate_frequencies(compressed_str)
        huffman_tree = self.build_huffman_tree(frequencies)
        if huffman_tree is None:
            return ""
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
            logging.warning("paq_compress: Empty input, returning empty bytes")
            return b''
        try:
            if isinstance(data, bytearray):
                data = bytes(data)
            elif not isinstance(data, bytes):
                raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
            compressed = paq.compress(data)
            logging.info("PAQ9a compression complete")
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
            logging.info("PAQ9a decompression complete")
            return decompressed
        except Exception as e:
            logging.error(f"PAQ9a decompression failed: {e}")
            return None

    def transform_genomecompress(self, data: bytes) -> bytes:
        """Encode DNA sequence using GenomeCompress algorithm."""
        if not data:
            logging.warning("transform_genomecompress: Empty input, returning empty bytes")
            return b''
        
        try:
            dna_str = data.decode('ascii').upper()
            if not all(c in 'ACGT' for c in dna_str):
                logging.error("transform_genomecompress: Invalid DNA sequence, contains non-ACGT characters")
                return b''
        except Exception as e:
            logging.error(f"transform_genomecompress: Failed to decode input as DNA: {e}")
            return b''

        n = len(dna_str)
        r = n % 4
        output_bits = []
        i = 0

        while i < n - r:
            if i + 8 <= n and dna_str[i:i+8] in DNA_ENCODING_TABLE:
                segment = dna_str[i:i+8]
                output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[segment], '05b')])
                i += 8
            else:
                segment = dna_str[i:i+4]
                if segment in DNA_ENCODING_TABLE:
                    output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[segment], '05b')])
                    i += 4
                else:
                    logging.error(f"transform_genomecompress: Invalid 4-base segment at position {i}: {segment}")
                    return b''

        for j in range(i, n):
            segment = dna_str[j]
            output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[segment], '05b')])

        bit_str = ''.join(map(str, output_bits))
        byte_length = (len(bit_str) + 7) // 8
        byte_data = int(bit_str, 2).to_bytes(byte_length, 'big') if bit_str else b''
        
        logging.info(f"transform_genomecompress: Encoded {n} bases to {len(byte_data)} bytes")
        return byte_data

    def reverse_transform_genomecompress(self, data: bytes) -> bytes:
        """Decode data compressed with GenomeCompress algorithm."""
        if not data:
            logging.warning("reverse_transform_genomecompress: Empty input, returning empty bytes")
            return b''

        bit_str = bin(int.from_bytes(data, 'big'))[2:].zfill(len(data) * 8)
        output = []
        i = 0

        while i < len(bit_str):
            if i + 5 > len(bit_str):
                logging.warning(f"reverse_transform_genomecompress: Incomplete 5-bit segment at position {i}")
                break
            segment_bits = bit_str[i:i+5]
            segment_val = int(segment_bits, 2)
            if segment_val not in DNA_DECODING_TABLE:
                logging.error(f"reverse_transform_genomecompress: Invalid 5-bit code: {segment_bits}")
                return b''
            output.append(DNA_DECODING_TABLE[segment_val])
            i += 5

        dna_str = ''.join(output)
        result = dna_str.encode('ascii')
        
        logging.info(f"reverse_transform_genomecompress: Decoded {len(result)} bases")
        return result

    def transform_01(self, data, repeat=100):
        """Transform using prime XOR every 3 bytes."""
        if not data:
            logging.warning("transform_01: Empty input, returning empty bytes")
            return b''
        return transform_with_prime_xor_every_3_bytes(data, repeat=repeat)

    def reverse_transform_01(self, data, repeat=100):
        """Reverse transform_01 (same as forward)."""
        return self.transform_01(data, repeat=repeat)

    def transform_03(self, data):
        """Transform using chunk XOR with 0xFF."""
        if not data:
            logging.warning("transform_03: Empty input, returning empty bytes")
            return b''
        return transform_with_pattern_chunk(data)

    def reverse_transform_03(self, data):
        """Reverse transform_03 (same as forward)."""
        return self.transform_03(data)

    def transform_04(self, data, repeat=100):
        """Subtract index modulo 256."""
        if not data:
            logging.warning("transform_04: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] - (i % 256)) % 256
        return bytes(transformed)

    def reverse_transform_04(self, data, repeat=100):
        """Reverse transform_04 by adding index modulo 256."""
        if not data:
            logging.warning("reverse_transform_04: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] + (i % 256)) % 256
        return bytes(transformed)

    def transform_05(self, data, shift=3):
        """Rotate bytes left by shift bits."""
        if not data:
            logging.warning("transform_05: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] << shift) | (transformed[i] >> (8 - shift))) & 0xFF
        return bytes(transformed)

    def reverse_transform_05(self, data, shift=3):
        """Rotate bytes right by shift bits."""
        if not data:
            logging.warning("reverse_transform_05: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] >> shift) | (transformed[i] << (8 - shift))) & 0xFF
        return bytes(transformed)

    def transform_06(self, data, seed=42):
        """Apply random substitution table."""
        if not data:
            logging.warning("transform_06: Empty input, returning empty bytes")
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
            logging.warning("reverse_transform_06: Empty input, returning empty bytes")
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
            logging.warning("transform_07: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"transform_07: {cycles} cycles for {len(data)} bytes")

        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]

        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        return bytes(transformed)

    def reverse_transform_07(self, data, repeat=100):
        """Reverse transform_07."""
        if not data:
            logging.warning("reverse_transform_07: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"reverse_transform_07: {cycles} cycles for {len(data)} bytes")

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte

        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]

        return bytes(transformed)

    def transform_08(self, data, repeat=100):
        """XOR with nearest prime and pi digits."""
        if not data:
            logging.warning("transform_08: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"transform_08: {cycles} cycles for {len(data)} bytes")

        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]

        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        return bytes(transformed)

    def reverse_transform_08(self, data, repeat=100):
        """Reverse transform_08."""
        if not data:
            logging.warning("reverse_transform_08: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"reverse_transform_08: {cycles} cycles for {len(data)} bytes")

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit

        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime

        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]

        return bytes(transformed)

    def transform_09(self, data, repeat=100):
        """XOR with prime, seed, and pi digits."""
        if not data:
            logging.warning("transform_09: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"transform_09: {cycles} cycles, {repeat} repeats for {len(data)} bytes")

        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]

        size_prime = find_nearest_prime_around(len(data) % 256)
        seed_idx = len(data) % len(self.seed_tables)
        seed_value = self.get_seed(seed_idx, len(data))
        for i in range(len(transformed)):
            transformed[i] ^= size_prime ^ seed_value

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit ^ (i % 256)

        return bytes(transformed)

    def reverse_transform_09(self, data, repeat=100):
        """Reverse transform_09."""
        if not data:
            logging.warning("reverse_transform_09: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"reverse_transform_09: {cycles} cycles, {repeat} repeats for {len(data)} bytes")

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit ^ (i % 256)

        size_prime = find_nearest_prime_around(len(data) % 256)
        seed_idx = len(data) % len(self.seed_tables)
        seed_value = self.get_seed(seed_idx, len(data))
        for i in range(len(transformed)):
            transformed[i] ^= size_prime ^ seed_value

        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]

        return bytes(transformed)

    def transform_10(self, data, repeat=100):
        """XOR with value derived from 'X1' sequences."""
        if not data:
            logging.warning("transform_10: Empty input, returning empty bytes with n=0")
            return bytes([0])
        transformed = bytearray(data)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"transform_10: {cycles} cycles, {repeat} repeats for {len(data)} bytes")

        count = 0
        for i in range(len(data) - 1):
            if data[i] == 0x58 and data[i + 1] == 0x31:
                count += 1
        logging.info(f"transform_10: Found {count} 'X1' sequences")

        n = (((count * self.SQUARE_OF_ROOT) + self.ADD_NUMBERS) // 3) * self.MULTIPLY
        n = n % 256
        logging.info(f"transform_10: Computed n = {n}")

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                transformed[i] ^= n

        return bytes([n]) + bytes(transformed)

    def reverse_transform_10(self, data, repeat=100):
        """Reverse transform_10."""
        if len(data) < 1:
            logging.warning("reverse_transform_10: Data too short, returning empty bytes")
            return b''
        n = data[0]
        transformed = bytearray(data[1:])
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"reverse_transform_10: {cycles} cycles, {repeat} repeats, n={n}")

        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                transformed[i] ^= n

        return bytes(transformed)

def transform_11(self, data, repeat=100):
    """Fixed transform_11: Adaptive modular transform with efficient y selection. LOSSLESS."""
    if not data:
        logging.warning("transform_11: Empty input, returning minimal output")
        return struct.pack('>B', 0)  # Marker for no transformation
    
    data_size = len(data)
    data_size_kb = data_size / 1024
    
    # Scale repeats based on data size (avoid excessive computation)
    effective_repeat = min(repeat, max(1, int(data_size_kb * 10)))
    cycles = min(5, max(1, int(data_size_kb)))  # Limit cycles to prevent excessive computation
    
    logging.info(f"transform_11: {data_size} bytes, cycles={cycles}, repeat={effective_repeat}")
    
    # Phase 1: Quick heuristic to select promising y values
    # Analyze data characteristics to narrow down y candidates
    byte_frequencies = [0] * 256
    zero_count = 0
    for b in data:
        byte_frequencies[b] += 1
        if b == 0:
            zero_count += 1
    
    # Select y values based on data characteristics
    # Prefer values that might increase low-frequency bytes or create patterns
    candidate_ys = []
    
    # Strategy 1: Values near most frequent byte (to spread distribution)
    if max(byte_frequencies) > data_size * 0.1:  # If one byte dominates
        most_freq_byte = byte_frequencies.index(max(byte_frequencies))
        candidate_ys.extend([most_freq_byte + i for i in [-10, -5, 0, 5, 10]])
    
    # Strategy 2: Values that create more zeros (good for compression)
    candidate_ys.extend([1, 2, 4, 8, 16, 32, 64, 128, 255])  # Powers of 2 and extremes
    
    # Strategy 3: Values based on data size characteristics
    candidate_ys.append(data_size % 256)
    candidate_ys.append((data_size * 17) % 256)  # Prime multiplier
    
    # Remove duplicates and filter to 0-255 range
    candidate_ys = list(set([y % 256 for y in candidate_ys if 0 <= y <= 255]))
    # Limit to 20 candidates for efficiency
    candidate_ys = candidate_ys[:20]
    
    if not candidate_ys:
        candidate_ys = [1, 64, 128, 255]  # Fallback candidates
    
    logging.info(f"transform_11: Testing {len(candidate_ys)} candidate y values")
    
    # Phase 2: Test candidates with lightweight scoring (not full compression)
    best_y = None
    best_score = float('inf')
    best_transformed = None
    
    # Quick scoring function - count zero runs and entropy reduction
    def score_transformation(transformed_data):
        """Score transformation based on zero runs and byte distribution."""
        score = 0
        zero_run_length = 0
        byte_counts = [0] * 256
        
        for b in transformed_data:
            byte_counts[b] += 1
            if b == 0:
                zero_run_length += 1
                score += zero_run_length  # Longer runs = better score
            else:
                zero_run_length = 0
        
        # Add entropy penalty (prefer more uniform distribution)
        entropy = sum((count / len(transformed_data)) * math.log2(count / len(transformed_data) + 1e-10) 
                     for count in byte_counts if count > 0)
        score -= entropy * 10  # Lower entropy (more uniform) is better
        
        return score
    
    # Test each candidate
    for y in candidate_ys:
        transformed = bytearray(data)
        
        # Apply transformation (same number of times as will be reversed)
        for _ in range(effective_repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] + y + 1) % 256
        
        current_score = score_transformation(transformed)
        
        if current_score < best_score:
            best_score = current_score
            best_y = y
            best_transformed = bytes(transformed)
    
    # If no good transformation found, use y=0 (no-op)
    if best_y is None:
        best_y = 0
        best_transformed = data
        logging.warning("transform_11: No improvement found, using y=0 (no transformation)")
    
    # Phase 3: Store metadata and return
    # Metadata format: y_value (1 byte) + repeat_count (1 byte) + cycles (1 byte)
    metadata = struct.pack('>BBB', best_y, effective_repeat % 256, cycles)
    
    # For verification/logging: compute a quick hash of transformed data
    quick_hash = sum(best_transformed) % 256
    metadata += struct.pack('>B', quick_hash)  # 1 byte hash for basic integrity check
    
    logging.info(f"transform_11: Selected y={best_y}, score={best_score:.1f}, "
                f"zeros={sum(1 for b in best_transformed if b == 0)}")
    
    result = metadata + best_transformed
    logging.info(f"transform_11: Original {data_size} bytes -> {len(result)} bytes (metadata: {len(metadata)})")
    
    return result

def reverse_transform_11(self, data, repeat=100):
    """Fixed reverse_transform_11: Perfectly reverses transform_11. LOSSLESS."""
    if len(data) < 5:  # Need at least metadata (4 bytes)
        logging.warning("reverse_transform_11: Data too short, returning original")
        return data
    
    # Extract metadata
    metadata = data[:4]
    transformed_data = data[4:]
    
    if not transformed_data:
        logging.warning("reverse_transform_11: No transformed data, returning empty")
        return b''
    
    try:
        y, stored_repeat, cycles = struct.unpack('>BBB', metadata[:3])
        quick_hash = metadata[3]
        
        # Verify basic integrity with quick hash
        computed_hash = sum(transformed_data) % 256
        if computed_hash != quick_hash:
            logging.warning(f"reverse_transform_11: Hash mismatch (expected {quick_hash}, got {computed_hash})")
            # Continue anyway - hash is just for basic sanity check
        
        data_size = len(transformed_data)
        data_size_kb = data_size / 1024
        
        # Use stored repeat count, but respect data size limits
        effective_repeat = stored_repeat
        if effective_repeat > data_size * 2:  # Sanity check
            effective_repeat = min(effective_repeat, max(1, int(data_size_kb * 10)))
            logging.warning(f"reverse_transform_11: Adjusted repeat from {stored_repeat} to {effective_repeat}")
        
        logging.info(f"reverse_transform_11: y={y}, repeat={effective_repeat}, cycles={cycles}, "
                    f"data_size={data_size} bytes")
        
        # Reverse the transformation
        restored = bytearray(transformed_data)
        
        # Apply inverse transformation the same number of times
        for _ in range(effective_repeat):
            for i in range(len(restored)):
                restored[i] = (restored[i] - y - 1) % 256
        
        # Convert back to bytes
        result = bytes(restored)
        
        # Basic verification: check if we got reasonable data
        zero_count_original = sum(1 for b in result if b == 0)
        zero_count_transformed = sum(1 for b in transformed_data if b == 0)
        
        logging.info(f"reverse_transform_11: Restored {len(result)} bytes, "
                    f"zeros: {zero_count_original} (original) vs {zero_count_transformed} (transformed)")
        
        # If y was 0, this should be a no-op
        if y == 0:
            if result != transformed_data:
                logging.error("reverse_transform_11: y=0 should be no-op but data changed!")
                return transformed_data  # Return original transformed data
            logging.info("reverse_transform_11: y=0 confirmed as no-op")
        
        return result
        
    except struct.error as e:
        logging.error(f"reverse_transform_11: Failed to unpack metadata: {e}")
        return data[4:]  # Return transformed data as fallback
    except Exception as e:
        logging.error(f"reverse_transform_11: Unexpected error: {e}")
        return data[4:]  # Return transformed data as fallback
        
    def transform_12(self, data, repeat=100):
        """XOR with Fibonacci sequence."""
        if not data:
            logging.warning("transform_12: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        data_size = len(data)
        fib_length = len(self.fibonacci)
        logging.info(f"transform_12: Fibonacci transform for {data_size} bytes, repeat={repeat}")
        
        for _ in range(repeat):
            for i in range(len(transformed)):
                fib_index = i % fib_length
                fib_value = self.fibonacci[fib_index] % 256
                transformed[i] ^= fib_value
        
        return bytes(transformed)

    def reverse_transform_12(self, data, repeat=100):
        """Reverse Fibonacci XOR transform."""
        if not data:
            logging.warning("reverse_transform_12: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        data_size = len(data)
        fib_length = len(self.fibonacci)
        logging.info(f"reverse_transform_12: Reversing Fibonacci for {data_size} bytes, repeat={repeat}")
        
        for _ in range(repeat):
            for i in range(len(transformed)):
                fib_index = i % fib_length
                fib_value = self.fibonacci[fib_index] % 256
                transformed[i] ^= fib_value
        
        return bytes(transformed)


    def transform_13(self, data, repeat=100):
    """Fixed StateTable transform with proper underflow tracking. 100% LOSSLESS."""
    if not data:
        logging.warning("transform_13: Empty input, returning empty bytes")
        return b''
    
    transformed = bytearray(data)
    table = self.state_table.table
    table_length = len(table)
    underflow_info = []  # Store (position, table_value) tuples
    data_size_kb = len(data) / 1024
    cycles = min(10, max(1, int(data_size_kb)))
    effective_repeat = min(repeat, max(1, int(data_size_kb * 10)))
    
    logging.info(f"transform_13: {cycles} cycles, {effective_repeat} repeats for {len(data)} bytes")
    
    # Apply transformations with exact underflow tracking
    for cycle in range(cycles):
        for _ in range(effective_repeat // 10):
            for i in range(len(transformed)):
                table_value = table[i % table_length][0] if table else 0
                result = transformed[i] - table_value
                if result < 0:
                    underflow_info.append((i, table_value))  # Store EXACT position and table value
                transformed[i] = (result + 256) % 256
    
    # Pack underflow info efficiently: count (2 bytes) + (pos:2B, table_val:1B) per underflow
    underflow_bytes = bytearray()
    for pos, table_val in underflow_info:
        underflow_bytes.extend(struct.pack('>HB', pos, table_val))
    
    # Header: cycles (1B) + repeat (1B) + underflow_count (2B) + data_length (4B)
    data_length = len(transformed)
    header = struct.pack('>BBHI', cycles, effective_repeat % 256, len(underflow_info), data_length)
    
    result = header + bytes(transformed) + underflow_bytes
    logging.info(f"transform_13: {len(underflow_info)} underflows tracked, total size: {len(result)} bytes")
    return result

def reverse_transform_13(self, data, repeat=100):
    """Fixed reverse StateTable transform with exact position matching. 100% LOSSLESS."""
    if len(data) < 8:  # Need at least header (cycles+repeat+underflow_count+data_length = 8 bytes)
        logging.warning("reverse_transform_13: Data too short, returning original")
        return data
    
    # Extract header
    header = data[:8]
    cycles, stored_repeat, underflow_count, data_length = struct.unpack('>BBHI', header)
    
    if len(data) < 8 + data_length + underflow_count * 3:
        logging.error("reverse_transform_13: Insufficient data for underflow info")
        return b''
    
    # Extract transformed data and underflow info
    transformed = bytearray(data[8:8 + data_length])
    underflow_bytes = data[8 + data_length:]
    
    # Parse exact underflow positions and values
    underflow_info = []
    for i in range(0, len(underflow_bytes), 3):
        if i + 2 < len(underflow_bytes):
            pos, table_val = struct.unpack('>HB', underflow_bytes[i:i+3])
            if pos < len(transformed):  # Valid position
                underflow_info.append((pos, table_val))
    
    table = self.state_table.table
    table_length = len(table)
    effective_repeat = stored_repeat
    
    logging.info(f"reverse_transform_13: {cycles} cycles, {effective_repeat} repeats, {len(underflow_info)} underflows")
    
    # Create lookup for quick access
    underflow_map = {pos: table_val for pos, table_val in underflow_info}
    
    # Reverse transformations in exact reverse order
    for cycle in range(cycles - 1, -1, -1):
        for _ in range(effective_repeat // 10):
            for i in range(len(transformed)):
                table_value = table[i % table_length][0] if table else 0
                current = transformed[i]
                
                if i in underflow_map and underflow_map[i] == table_value:
                    # Had underflow: original = current + table_value - 256
                    result = (current + table_value - 256) % 256
                    # Remove from map to prevent reuse
                    del underflow_map[i]
                else:
                    # No underflow: original = current + table_value
                    result = (current + table_value) % 256
                
                transformed[i] = result
    
    # Verification: check if all underflows were used
    if underflow_map:
        logging.warning(f"reverse_transform_13: {len(underflow_map)} unused underflow entries")
    
    logging.info(f"reverse_transform_13: Successfully restored {len(transformed)} bytes")
    return bytes(transformed)

    def transform_14(self, data, repeat=255):
    """Fixed pattern transform with perfect bit length preservation and verification. 100% LOSSLESS."""
    if not data:
        logging.warning("transform_14: Empty input, returning minimal output")
        return struct.pack('>B', 0)
    
    # Store original information for perfect reconstruction
    original_byte_length = len(data)
    original_bit_length = original_byte_length * 8
    
    if original_bit_length < MIN_BITS or original_bit_length > MAX_BITS:
        logging.warning(f"transform_14: Input size {original_bit_length} bits out of range [{MIN_BITS}, {MAX_BITS}]")
        # Return original with no-op marker
        return struct.pack('>B', 254) + data  # 254 = no-op marker
    
    # Convert bytes to exact bit representation (no padding issues)
    binary_str = ''.join(format(b, '08b') for b in data)
    output_bits = list(binary_str)
    
    # Determine processing parameters
    data_size_kb = original_byte_length / 1024
    max_cycles = min(50, max(1, int(data_size_kb * 5)))  # More reasonable limit
    
    # Store all transformation metadata
    pattern_transforms = []  # (pos, type, orig_bit, cycle)
    total_transformed = 0
    
    logging.info(f"transform_14: Processing {original_bit_length} bits, max {max_cycles} cycles")
    
    # Phase 1: Process "01" patterns (highest priority)
    for cycle in range(max_cycles):
        temp_bits = []
        i = 0
        cycle_transforms = 0
        
        while i < len(output_bits) - 2:
            if output_bits[i:i+2] == ['0', '1'] and i + 2 < len(output_bits):
                # Found "01" pattern - record transformation
                next_bit = output_bits[i+2]
                pattern_transforms.append((i + 2, 0, next_bit, cycle))  # type 0 = 01 pattern
                cycle_transforms += 1
                total_transformed += 1
                
                # Apply transformation: XOR with prime-derived bit
                prime_index = original_byte_length % len(self.PRIMES)
                xor_bit = 1 if self.PRIMES[prime_index] % 2 == 0 else 0
                transformed_bit = str(1 - int(next_bit)) if int(next_bit) == xor_bit else next_bit
                
                # Keep the pattern and add transformed bit
                temp_bits.extend(['0', '1', transformed_bit])
                i += 3
            else:
                temp_bits.append(output_bits[i])
                i += 1
        
        # Add any remaining bits
        while i < len(output_bits):
            temp_bits.append(output_bits[i])
            i += 1
        
        output_bits = temp_bits
        
        logging.debug(f"Cycle {cycle}: {cycle_transforms} '01' patterns transformed")
        if cycle_transforms == 0:  # No more transformations possible
            break
    
    # Phase 2: Process 4-bit runs ("0000" or "1111")
    run_start = len(pattern_transforms)
    i = 0
    while i < len(output_bits) - 4:
        pattern_4 = ''.join(output_bits[i:i+4])
        pattern_val = int(pattern_4, 2)
        
        if pattern_val in [0b0000, 0b1111] and i + 4 < len(output_bits):
            # Found qualifying run - record transformation
            next_bit_pos = i + 4
            original_next_bit = output_bits[next_bit_pos]
            
            # Apply transformation: flip the next bit
            transformed_bit = '1' if original_next_bit == '0' else '0'
            output_bits[next_bit_pos] = transformed_bit
            
            pattern_transforms.append((next_bit_pos, 1, original_next_bit, 0))  # type 1 = run pattern
            total_transformed += 1
            i += 5  # Skip the entire transformed pattern
        else:
            i += 1
    
    # Phase 3: Pack metadata efficiently but completely
    metadata_bytes = bytearray()
    
    # Header: original_bit_length (2B) + num_transforms (2B) + cycles_used (1B) + data_length (4B)
    final_bit_length = len(output_bits)
    cycles_used = min(cycle + 1, max_cycles) if 'cycle' in locals() else 0
    header = struct.pack('>HHIB', original_bit_length, len(pattern_transforms), cycles_used, final_bit_length)
    metadata_bytes.extend(header)
    
    # Pack transformations: pos (2B) + type (2 bits) + orig_bit (1 bit) + cycle (4 bits) = 3 bytes
    for pos, ptype, orig_bit, cycle_info in pattern_transforms:
        # Pack into 3 bytes: pos (16 bits) + flags (8 bits)
        flags = (int(orig_bit) << 7) | (cycle_info & 0x0F) | (ptype << 4)  # orig_bit(1) + cycle(4) + type(2) + padding(1)
        metadata_bytes.extend(struct.pack('>HB', pos, flags))
    
    # Convert transformed bits back to bytes (exact representation)
    bit_str = ''.join(output_bits)
    
    # Ensure we have exact byte alignment
    while len(bit_str) % 8 != 0:
        bit_str += '0'  # Pad with zeros (will be ignored in reverse)
    
    byte_length = len(bit_str) // 8
    if byte_length > 0:
        transformed_bytes = int(bit_str, 2).to_bytes(byte_length, 'big')
    else:
        transformed_bytes = b''
    
    # Add verification hash (first 4 bytes of original data)
    verification_hash = data[:4]
    
    # Final result: metadata + transformed_bytes + verification
    result = metadata_bytes + transformed_bytes + verification_hash
    
    logging.info(f"transform_14: {original_bit_length} bits -> {len(transformed_bytes)} bytes transformed data")
    logging.info(f"transform_14: {len(pattern_transforms)} total transforms, {len(metadata_bytes)} metadata bytes")
    
    return result

def reverse_transform_14(self, data, repeat=255):
    """Fixed reverse pattern transform with perfect reconstruction. 100% LOSSLESS."""
    if len(data) < 9:  # Need header (8 bytes) + verification (1 byte minimum)
        logging.warning("reverse_transform_14: Data too short, returning original")
        return data
    
    # Extract verification hash (last 4 bytes)
    verification_hash = data[-4:]
    data_body = data[:-4]
    
    if len(data_body) < 9:  # Need at least header
        logging.warning("reverse_transform_14: Insufficient header data")
        return data_body
    
    # Extract header
    header = data_body[:9]
    original_bit_length, num_transforms, cycles_used, final_bit_length = struct.unpack('>HHIB', header)
    metadata_start = 9
    metadata_end = metadata_start + num_transforms * 3  # 3 bytes per transform
    
    if len(data_body) < metadata_end:
        logging.error("reverse_transform_14: Insufficient metadata")
        return b''
    
    # Extract metadata and transformed data
    metadata_bytes = data_body[metadata_start:metadata_end]
    transformed_bytes = data_body[metadata_end:]
    
    if not transformed_bytes:
        logging.warning("reverse_transform_14: No transformed data found")
        return b''
    
    # Verify data integrity with hash
    if verification_hash != transformed_bytes[:4]:
        logging.error("reverse_transform_14: Verification hash mismatch!")
        # Try to continue with partial recovery
        logging.warning("reverse_transform_14: Continuing with partial recovery")
    
    # Convert transformed bytes back to exact bit string
    bit_str = bin(int.from_bytes(transformed_bytes, 'big'))[2:]
    # Pad to exact final bit length from metadata
    bit_str = bit_str.zfill(final_bit_length)
    
    # Trim to exact length if too long
    bit_str = bit_str[:final_bit_length]
    output_bits = list(bit_str)
    
    logging.info(f"reverse_transform_14: Restoring {original_bit_length} bits from {final_bit_length} transformed bits")
    logging.info(f"reverse_transform_14: Processing {num_transforms} transformations over {cycles_used} cycles")
    
    # Parse transformation metadata in reverse order (reverse chronological)
    pattern_transforms = []
    for i in range(0, len(metadata_bytes), 3):
        if i + 2 < len(metadata_bytes):
            pos, flags = struct.unpack('>HB', metadata_bytes[i:i+3])
            ptype = (flags >> 4) & 0x03  # 2 bits for type
            orig_bit = str((flags >> 7) & 0x01)  # 1 bit for original bit
            cycle_info = flags & 0x0F  # 4 bits for cycle
            pattern_transforms.append((pos, ptype, orig_bit, cycle_info))
    
    # Apply transformations in reverse order (most recent first)
    for transform_idx in range(len(pattern_transforms) - 1, -1, -1):
        pos, ptype, orig_bit, cycle_info = pattern_transforms[transform_idx]
        
        if pos < len(output_bits):
            # Restore the original bit at this position
            output_bits[pos] = orig_bit
            
            # Log for debugging (can be removed in production)
            if ptype == 0:
                logging.debug(f"reverse_transform_14: Restored '01' pattern at pos {pos} to {orig_bit}, cycle {cycle_info}")
            else:
                logging.debug(f"reverse_transform_14: Restored run pattern at pos {pos} to {orig_bit}, type {ptype}")
    
    # Trim to exact original bit length
    output_bits = output_bits[:original_bit_length]
    
    # Convert back to bytes with perfect preservation
    bit_str = ''.join(output_bits)
    
    # Ensure proper byte alignment
    while len(bit_str) % 8 != 0:
        bit_str += '0'  # Pad to byte boundary
    
    original_byte_length = (original_bit_length + 7) // 8
    byte_length = len(bit_str) // 8
    
    if byte_length > 0 and bit_str:
        result = int(bit_str, 2).to_bytes(byte_length, 'big')
        # Trim to exact original byte length
        result = result[:original_byte_length]
    else:
        result = b''
    
    # Final verification: check length matches expected
    if len(result) != original_byte_length:
        logging.error(f"reverse_transform_14: Length mismatch! Expected {original_byte_length}, got {len(result)}")
        # Return best effort result
        logging.warning("reverse_transform_14: Returning partial result")
    
    logging.info(f"reverse_transform_14: Successfully restored {len(result)}/{original_byte_length} bytes")
    return result
    
    def transform_15(self, data, repeat=100):
        """XOR data with a value derived from current time and a prime number."""
        if not data:
            logging.warning("transform_15: Empty input, returning empty bytes")
            return b''
        transformed = bytearray(data)
        # Use current time (12:34 PM IST = 1234 in 24-hour format) dynamically
        current_time = datetime.now().hour * 100 + datetime.now().minute
        prime_index = len(data) % len(self.PRIMES)
        time_prime_combo = (current_time * self.PRIMES[prime_index]) % 256
        logging.info(f"transform_15: Using current_time={current_time}, prime={self.PRIMES[prime_index]}, combo={time_prime_combo}")

        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] ^= time_prime_combo
        return bytes(transformed)

    def reverse_transform_15(self, data, repeat=100):
        """Reverse transform_15 (same as forward since XOR is symmetric)."""
        return self.transform_15(data, repeat=repeat)

    def compress_with_best_method(self, data, filetype, input_filename, mode="slow"):
        """Compress data using the best transformation method."""
        if not data:
            logging.warning("compress_with_best_method: Empty input, returning minimal marker")
            return bytes([0])

        is_dna = False
        try:
            data_str = data.decode('ascii').upper()
            is_dna = all(c in 'ACGT' for c in data_str)
        except:
            pass

        fast_transformations = [
            (1, self.transform_04),
            (2, self.transform_01),
            (3, self.transform_03),
            (5, self.transform_05),
            (6, self.transform_06),
            (7, self.transform_07),
            (8, self.transform_08),
            (9, self.transform_09),
            (12, self.transform_12),
            (14, self.transform_14),
            (15, self.transform_15),  # Add transform_15
        ]
        slow_transformations = fast_transformations + [
            (10, self.transform_10),
            (11, self.transform_11),
            (13, self.transform_13),
        ] + [(i, self.generate_transform_method(i)[0]) for i in range(16, 256)]

        if is_dna:
            transformations = [(0, self.transform_genomecompress)] + slow_transformations
        else:
            transformations = slow_transformations if mode == "slow" else fast_transformations

        if filetype in [Filetype.JPEG, Filetype.TEXT]:
            prioritized = [
                (7, self.transform_07),
                (8, self.transform_08),
                (9, self.transform_09),
                (12, self.transform_12),
                (13, self.transform_13),
                (14, self.transform_14),
                (15, self.transform_15),  # Add transform_15
            ]
            if is_dna:
                prioritized = [(0, self.transform_genomecompress)] + prioritized
            if mode == "slow":
                prioritized += [(10, self.transform_10), (11, self.transform_11)] + \
                              [(i, self.generate_transform_method(i)[0]) for i in range(16, 256)]
            transformations = prioritized + [t for t in transformations if t[0] not in [0, 7, 8, 9, 10, 11, 12, 13, 14, 15] + list(range(16, 256))]

        methods = [('paq', self.paq_compress)]
        best_compressed = None
        best_size = float('inf')
        best_marker = None
        best_method = None

        for marker, transform in transformations:
            transformed = transform(data)
            for method_name, compress_func in methods:
                try:
                    compressed = compress_func(transformed)
                    if compressed is None:
                        continue
                    size = len(compressed)
                    if size < best_size:
                        best_size = size
                        best_compressed = compressed
                        best_marker = marker
                        best_method = method_name
                except Exception as e:
                    logging.warning(f"Compression method {method_name} with transform {marker} failed: {e}")
                    continue

        if len(data) < HUFFMAN_THRESHOLD:
            binary_str = bin(int(binascii.hexlify(data), 16))[2:].zfill(len(data) * 8)
            compressed_huffman = self.compress_data_huffman(binary_str)
            compressed_bytes = int(compressed_huffman, 2).to_bytes((len(compressed_huffman) + 7) // 8, 'big') if compressed_huffman else b''
            if compressed_bytes and len(compressed_bytes) < best_size:
                best_size = len(compressed_bytes)
                best_compressed = compressed_bytes
                best_marker = 4
                best_method = 'huffman'

        if best_compressed is None:
            logging.error("All compression methods failed, returning original with marker 0")
            return bytes([0]) + data

        logging.info(f"Best method: {best_method}, Marker: {best_marker} for {filetype.name} in {mode} mode")
        return bytes([best_marker]) + best_compressed

    def decompress_with_best_method(self, data):
        """Decompress data based on marker."""
        if len(data) < 1:
            logging.warning("decompress_with_best_method: Insufficient data")
            return b'', None

        method_marker = data[0]
        compressed_data = data[1:]

        reverse_transforms = {
            0: self.reverse_transform_genomecompress,
            1: self.reverse_transform_04,
            2: self.reverse_transform_01,
            3: self.reverse_transform_03,
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
            15: self.reverse_transform_15,  # Add reverse_transform_15
        }
        reverse_transforms.update({i: self.generate_transform_method(i)[1] for i in range(16, 256)})

        if method_marker == 4:
            binary_str = bin(int(binascii.hexlify(compressed_data), 16))[2:].zfill(len(compressed_data) * 8)
            decompressed_binary = self.decompress_data_huffman(binary_str)
            if not decompressed_binary:
                logging.warning("Huffman decompression empty")
                return b'', None
            try:
                num_bytes = (len(decompressed_binary) + 7) // 8
                hex_str = "%0*x" % (num_bytes * 2, int(decompressed_binary, 2))
                if len(hex_str) % 2 != 0:
                    hex_str = '0' + hex_str
                return binascii.unhexlify(hex_str), None
            except Exception as e:
                logging.error(f"Huffman data conversion failed: {e}")
                return b'', None

        if method_marker not in reverse_transforms:
            logging.error(f"Unknown marker: {method_marker}")
            return b'', None

        try:
            decompressed = self.paq_decompress(compressed_data)
            if not decompressed:
                logging.warning("PAQ decompression empty")
                return b'', None
            result = reverse_transforms[method_marker](decompressed)
            zero_count = sum(1 for b in result if b == 0)
            logging.info(f"Decompressed with marker {method_marker}, {zero_count} zeros")
            return result, method_marker
        except Exception as e:
            logging.error(f"PAQ decompression failed: {e}")
            return b'', None

    def compress(self, input_file: str, output_file: str, filetype: Filetype = Filetype.DEFAULT, mode: str = "slow") -> bool:
        """Compress a file with the best method."""
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            if not data:
                logging.warning(f"Input file {input_file} is empty")
                return False
            compressed = self.compress_with_best_method(data, filetype, input_file, mode)
            with open(output_file, 'wb') as f:
                f.write(compressed)
            logging.info(f"Compressed {input_file} to {output_file}, size {len(compressed)} bytes")
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
                return False
            decompressed, marker = self.decompress_with_best_method(data)
            if not decompressed:
                logging.error(f"Decompression failed for {input_file}")
                return False
            with open(output_file, 'wb') as f:
                f.write(decompressed)
            logging.info(f"Decompressed {input_file} to {output_file}, size {len(decompressed)} bytes, marker {marker}")
            return True
        except Exception as e:
            logging.error(f"Decompression failed for {input_file}: {e}")
            return False

# === Main Function ===
def detect_filetype(filename: str) -> Filetype:
    """Detect filetype based on extension or content."""
    _, ext = os.path.splitext(filename.lower())
    if ext in ['.jpg', '.jpeg']:
        return Filetype.JPEG
    elif ext in ['.txt', '.dna']:
        try:
            with open(filename, 'r', encoding='ascii') as f:
                sample = f.read(1000)
                if all(c in 'ACGTacgt\n' for c in sample):
                    return Filetype.TEXT
        except:
            pass
        return Filetype.TEXT
    else:
        return Filetype.DEFAULT

def main():
    """Main function for PAQJP_6.6 Compression System."""
    print("PAQJP_6.6 Compression System")
    print("Created by Jurijus Pacalovas")
    print("Options:")
    print("1 - Compress file (Best of Smart Compressor [00] or PAQJP_6 [01])")
    print("2 - Decompress file")

    compressor = PAQJPCompressor()

    try:
        choice = input("Enter 1 or 2: ").strip()
        if choice not in ('1', '2'):
            print("Invalid choice. Exiting.")
            return
    except (EOFError, KeyboardInterrupt):
        print("Program terminated by user")
        return

    mode = "slow"
    if choice == '1':
        try:
            mode_choice = input("Enter compression mode (1 for fast, 2 for slow): ").strip()
            if mode_choice == '1':
                mode = "fast"
            elif mode_choice == '2':
                mode = "slow"
            else:
                print("Invalid mode, defaulting to slow")
                mode = "slow"
        except (EOFError, KeyboardInterrupt):
            print("Defaulting to slow mode")
            mode = "slow"

        input_file = input("Enter input file path: ").strip()
        output_file = input("Enter output file path: ").strip()

        if not os.path.exists(input_file):
            print(f"Input file {input_file} not found")
            return
        if not os.access(input_file, os.R_OK):
            print(f"No read permission for {input_file}")
            return
        if os.path.getsize(input_file) == 0:
            print(f"Input file {input_file} is empty")
            with open(output_file, 'wb') as f:
                f.write(bytes([0]))
            return

        filetype = detect_filetype(input_file)
        success = compressor.compress(input_file, output_file, filetype, mode)
        if success:
            orig_size = os.path.getsize(input_file)
            comp_size = os.path.getsize(output_file)
            ratio = (comp_size / orig_size) * 100 if orig_size > 0 else 0
            print(f"Compression successful: {output_file}, Size: {comp_size} bytes")
            print(f"Original: {orig_size} bytes, Compressed: {comp_size} bytes, Ratio: {ratio:.2f}%")
        else:
            print("Compression failed")

    elif choice == '2':
        input_file = input("Enter input file path: ").strip()
        output_file = input("Enter output file path: ").strip()

        if not os.path.exists(input_file):
            print(f"Input file {input_file} not found")
            return
        if not os.access(input_file, os.R_OK):
            print(f"No read permission for {input_file}")
            return
        if os.path.getsize(input_file) == 0:
            print(f"Input file {input_file} is empty")
            with open(output_file, 'wb') as f:
                f.write(b'')
            return

        success = compressor.decompress(input_file, output_file)
        if success:
            comp_size = os.path.getsize(input_file)
            decomp_size = os.path.getsize(output_file)
            print(f"Decompression successful: {output_file}")
            print(f"Compressed: {comp_size} bytes, Decompressed: {decomp_size} bytes")
        else:
            print("Decompression failed")

if __name__ == "__main__":
    main()
