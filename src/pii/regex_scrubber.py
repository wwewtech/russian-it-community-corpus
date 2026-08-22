"""
Deterministic Regex PII Scrubber for Phones, Emails, Crypto Wallets, API Keys, IPs, and Credentials.
"""

import re


class RegexPIIScrubber:
    """
    High-precision Regex scrubber for identifying and redacting sensitive data.
    """

    def __init__(self):
        # 1. Phone Numbers (RU + International)
        # Matches +7, 8, +1, +380, +995, +374 etc. with standard delimiters
        self.phone_pattern = re.compile(
            r"(?<![\w=])(?:\+?[78]|\+?380|\+?995|\+?374|\+?998|\+?375|\+?1)"
            r"[\s\-\(]*(\d[\s]*\d[\s]*\d)[\s\-\)]*(\d[\s]*\d[\s]*\d)[\s\-]*(\d[\s]*\d)[\s\-]*(\d[\s]*\d)\b"
            r"|(?<![\w=])(?:\+\d{1,3}[\s\-]?)?\(?\d{3,4}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",
            re.IGNORECASE,
        )

        # 2. Email Addresses
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

        # 3. Cryptocurrency Wallets
        # Bitcoin (P2PKH, P2SH, Bech32)
        self.btc_pattern = re.compile(
            r"\b(?:1[a-km-zA-HJ-NP-Z1-9]{25,34}|3[a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59})\b"
        )
        # Ethereum / EVM (0x...)
        self.eth_pattern = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
        # TRON / TRC20 (T...)
        self.tron_pattern = re.compile(r"\bT[A-Za-z1-9]{33}\b")
        # TON (EQ..., UQ...)
        self.ton_pattern = re.compile(r"\b(?:EQ|UQ)[a-zA-Z0-9_-]{46}\b")

        # 4. API Keys and Tokens
        # OpenAI keys: sk-..., sk-proj-...
        self.openai_key_pattern = re.compile(r"\bsk-(?:proj-)?[a-zA-Z0-9_\-]{20,}\b")
        # GitHub Personal Access Tokens
        self.github_token_pattern = re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{20,50}\b|\bgithub_pat_[a-zA-Z0-9_]{30,}\b"
        )
        # Telegram Bot Tokens: 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
        self.tg_bot_token_pattern = re.compile(r"\b\d{8,12}:[a-zA-Z0-9_\-]{25,45}\b")
        # AWS Access Key ID
        self.aws_key_pattern = re.compile(r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b")
        # Generic Secret / API token assignments: api_key = "...", secret = "..."
        self.secret_assignment_pattern = re.compile(
            r"(?i)(?:api_key|apikey|secret_key|private_key|auth_token|client_secret|password|access_token)"
            r'\s*[:=]\s*["\']([a-zA-Z0-9_\-\.]{12,})["\']'
        )
        # JWT Token pattern
        self.jwt_pattern = re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b")
        # Private RSA / OpenSSH keys
        self.ssh_key_pattern = re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----")

        # 5. IPv4 Addresses (Mask external public IPs, keep loopback / generic version numbers)
        self.ipv4_pattern = re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        )

        # 6. Telegram Private Invite Links & Join Links
        self.tg_invite_pattern = re.compile(r"t\.me/(?:\+|joinchat/)[a-zA-Z0-9_\-]+", re.IGNORECASE)

        # 7. User Mentions (@username)
        self.mention_pattern = re.compile(r"(?<!\w)@([a-zA-Z0-9_]{4,32})\b")

    def scrub(self, text: str, mention_map: dict[str, str] = None) -> tuple[str, dict[str, int]]:
        """
        Scrub sensitive PII from text using regex patterns.
        Returns cleaned text and a dictionary with count of scrubbed entities.
        """
        if not text:
            return "", {}

        stats: dict[str, int] = {
            "phones": 0,
            "emails": 0,
            "crypto_wallets": 0,
            "api_keys": 0,
            "jwt_tokens": 0,
            "ssh_keys": 0,
            "ip_addresses": 0,
            "private_invites": 0,
            "user_mentions": 0,
        }

        # 1. SSH Keys
        def _sub_ssh(m):
            stats["ssh_keys"] += 1
            return "[PRIVATE_KEY_REDACTED]"

        text = self.ssh_key_pattern.sub(_sub_ssh, text)

        # 2. API Keys & Bot Tokens
        def _sub_openai(m):
            stats["api_keys"] += 1
            return "[API_KEY_REDACTED]"

        text = self.openai_key_pattern.sub(_sub_openai, text)

        def _sub_gh(m):
            stats["api_keys"] += 1
            return "[API_KEY_REDACTED]"

        text = self.github_token_pattern.sub(_sub_gh, text)

        def _sub_tg_bot(m):
            stats["api_keys"] += 1
            return "[BOT_TOKEN_REDACTED]"

        text = self.tg_bot_token_pattern.sub(_sub_tg_bot, text)

        def _sub_aws(m):
            stats["api_keys"] += 1
            return "[AWS_KEY_REDACTED]"

        text = self.aws_key_pattern.sub(_sub_aws, text)

        def _sub_jwt(m):
            stats["jwt_tokens"] += 1
            return "[JWT_TOKEN_REDACTED]"

        text = self.jwt_pattern.sub(_sub_jwt, text)

        def _sub_secret(m):
            stats["api_keys"] += 1
            return m.group(0).replace(m.group(1), "[SECRET_REDACTED]")

        text = self.secret_assignment_pattern.sub(_sub_secret, text)

        # 3. Crypto Wallets
        def _sub_btc(m):
            stats["crypto_wallets"] += 1
            return "[CRYPTO_WALLET_BTC]"

        text = self.btc_pattern.sub(_sub_btc, text)

        def _sub_eth(m):
            stats["crypto_wallets"] += 1
            return "[CRYPTO_WALLET_ETH]"

        text = self.eth_pattern.sub(_sub_eth, text)

        def _sub_tron(m):
            stats["crypto_wallets"] += 1
            return "[CRYPTO_WALLET_TRON]"

        text = self.tron_pattern.sub(_sub_tron, text)

        def _sub_ton(m):
            stats["crypto_wallets"] += 1
            return "[CRYPTO_WALLET_TON]"

        text = self.ton_pattern.sub(_sub_ton, text)

        # 4. Email Addresses
        def _sub_email(m):
            stats["emails"] += 1
            return "[EMAIL_REDACTED]"

        text = self.email_pattern.sub(_sub_email, text)

        # 5. Phone Numbers
        def _sub_phone(m):
            # Check if it looks like a version number like 1.2.3 or 10.0.19045
            match_str = m.group(0)
            if re.match(r"^\d+\.\d+\.\d+$", match_str.strip()):
                return match_str
            stats["phones"] += 1
            return "[PHONE_REDACTED]"

        text = self.phone_pattern.sub(_sub_phone, text)

        # 6. IPv4 Addresses (filter out loopback 127.0.0.1 and 0.0.0.0)
        def _sub_ip(m):
            ip = m.group(0)
            if ip in ("127.0.0.1", "0.0.0.0", "1.1.1.1", "8.8.8.8", "8.8.4.4"):
                return ip
            # Filter out common semantic versions like 3.10.4
            parts = [int(p) for p in ip.split(".")]
            if parts[0] == 0 or (parts[0] < 10 and parts[1] < 10 and parts[2] < 10 and parts[3] < 10):
                return ip
            stats["ip_addresses"] += 1
            return "[IP_REDACTED]"

        text = self.ipv4_pattern.sub(_sub_ip, text)

        # 7. Private Telegram invite links
        def _sub_inv(m):
            stats["private_invites"] += 1
            return "t.me/[INVITE_LINK_REDACTED]"

        text = self.tg_invite_pattern.sub(_sub_inv, text)

        # 8. User Mentions
        if mention_map is not None:

            def _sub_mention(m):
                username = m.group(1).lower()
                stats["user_mentions"] += 1
                if username in mention_map:
                    return f"@{mention_map[username]}"
                else:
                    return f"@{mention_map.setdefault(username, f'user_{len(mention_map) + 1}')}"

            text = self.mention_pattern.sub(_sub_mention, text)
        else:

            def _sub_mention_anon(m):
                stats["user_mentions"] += 1
                return "@user_anon"

            text = self.mention_pattern.sub(_sub_mention_anon, text)

        return text, stats
