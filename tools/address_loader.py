#!/usr/bin/env python3
"""
Advanced Bitcoin Miner - Address Loader Tool
Loads and validates Bitcoin addresses for the bloom filter
"""

import argparse
import sys
import os
import hashlib
import base58
import bech32
import re
from typing import List, Set, Tuple

class AddressLoader:
    def __init__(self):
        self.valid_addresses = set()
        self.invalid_addresses = set()
        self.stats = {
            'total_processed': 0,
            'valid_p2pkh': 0,
            'valid_p2sh': 0,
            'valid_bech32': 0,
            'duplicates': 0
        }
    
    def validate_p2pkh(self, address: str) -> bool:
        """Validate P2PKH address (starts with 1)"""
        if not address.startswith('1'):
            return False
        
        try:
            decoded = base58.b58decode_check(address)
            return len(decoded) == 21 and decoded[0] == 0x00
        except:
            return False
    
    def validate_p2sh(self, address: str) -> bool:
        """Validate P2SH address (starts with 3)"""
        if not address.startswith('3'):
            return False
        
        try:
            decoded = base58.b58decode_check(address)
            return len(decoded) == 21 and decoded[0] == 0x05
        except:
            return False
    
    def validate_bech32(self, address: str) -> bool:
        """Validate Bech32 address (starts with bc1)"""
        if not address.startswith('bc1'):
            return False
        
        try:
            hrp, data = bech32.bech32_decode(address)
            return hrp == 'bc' and data is not None
        except:
            return False
    
    def validate_address(self, address: str) -> Tuple[bool, str]:
        """Validate a Bitcoin address and return its type"""
        address = address.strip()
        
        if not address:
            return False, "empty"
        
        if self.validate_p2pkh(address):
            return True, "p2pkh"
        elif self.validate_p2sh(address):
            return True, "p2sh"
        elif self.validate_bech32(address):
            return True, "bech32"
        else:
            return False, "invalid"
    
    def load_from_file(self, filename: str) -> Set[str]:
        """Load addresses from file"""
        addresses = set()
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    address = line.strip()
                    if not address:
                        continue
                    
                    self.stats['total_processed'] += 1
                    
                    # بررسی duplicate
                    if address in addresses:
                        self.stats['duplicates'] += 1
                        continue
                    
                    # اعتبارسنجی
                    is_valid, addr_type = self.validate_address(address)
                    
                    if is_valid:
                        addresses.add(address)
                        if addr_type == "p2pkh":
                            self.stats['valid_p2pkh'] += 1
                        elif addr_type == "p2sh":
                            self.stats['valid_p2sh'] += 1
                        elif addr_type == "bech32":
                            self.stats['valid_bech32'] += 1
                    else:
                        self.invalid_addresses.add(address)
                        
                    # نمایش پیشرفت
                    if line_num % 10000 == 0:
                        print(f"Processed {line_num} addresses...")
                        
        except FileNotFoundError:
            print(f"❌ Error: File '{filename}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
        
        return addresses
    
    def save_valid_addresses(self, filename: str):
        """Save valid addresses to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for address in sorted(self.valid_addresses):
                    f.write(f"{address}\n")
            print(f"✅ Saved {len(self.valid_addresses)} valid addresses to {filename}")
        except Exception as e:
            print(f"❌ Error saving addresses: {e}")
    
    def print_statistics(self):
        """Print loading statistics"""
        print("\n📊 Address Loading Statistics:")
        print(f"   Total processed:    {self.stats['total_processed']:,}")
        print(f"   Valid P2PKH:        {self.stats['valid_p2pkh']:,}")
        print(f"   Valid P2SH:         {self.stats['valid_p2sh']:,}")
        print(f"   Valid Bech32:       {self.stats['valid_bech32']:,}")
        print(f"   Duplicates:         {self.stats['duplicates']:,}")
        print(f"   Invalid addresses:  {len(self.invalid_addresses):,}")
        print(f"   Total valid:        {len(self.valid_addresses):,}")

def main():
    parser = argparse.ArgumentParser(description='Bitcoin Address Loader')
    parser.add_argument('input_file', help='Input file containing Bitcoin addresses')
    parser.add_argument('-o', '--output', default='data/addresses/valid_addresses.txt',
                       help='Output file for valid addresses')
    parser.add_argument('--invalid', help='Output file for invalid addresses')
    
    args = parser.parse_args()
    
    # بررسی وجود فایل ورودی
    if not os.path.exists(args.input_file):
        print(f"❌ Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # ایجاد دایرکتوری خروجی
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("🚀 Bitcoin Address Loader")
    print(f"📁 Input:  {args.input_file}")
    print(f"📁 Output: {args.output}")
    
    # لود آدرس‌ها
    loader = AddressLoader()
    valid_addresses = loader.load_from_file(args.input_file)
    loader.valid_addresses = valid_addresses
    
    # ذخیره آدرس‌های معتبر
    loader.save_valid_addresses(args.output)
    
    # ذخیره آدرس‌های نامعتبر (اگر درخواست شده)
    if args.invalid and loader.invalid_addresses:
        try:
            with open(args.invalid, 'w', encoding='utf-8') as f:
                for address in sorted(loader.invalid_addresses):
                    f.write(f"{address}\n")
            print(f"📝 Saved {len(loader.invalid_addresses)} invalid addresses to {args.invalid}")
        except Exception as e:
            print(f"⚠️ Could not save invalid addresses: {e}")
    
    # نمایش آمار
    loader.print_statistics()
    
    print("\n✅ Address loading completed!")

if __name__ == "__main__":
    main()