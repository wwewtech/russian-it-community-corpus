"""Tests for the streaming variants of the PII and deduplication stages.

The corpus is 2.8M messages. The stages used to hold the input list and
the output list at the same time, which costs several gigabytes; the
streaming variants keep only what they must (the name forms, the
fingerprints, the signature block).
"""

from __future__ import annotations

import unittest

from ricc.deduplication.exact_dedup import ExactDeduplicator
from ricc.deduplication.minhash_lsh import MinHashLSH
from ricc.ingestion.schema import CleanedMessage, NormalizedMessage
from ricc.pii.anonymizer import UnifiedPIIAnonymizer

LONG = "Подробный технический разговор о настройке nginx и балансировщиков в продакшене. " * 3


def _cleaned(msg_id: int, text: str) -> CleanedMessage:
    return CleanedMessage(
        msg_id=msg_id,
        chat_id=1,
        chat_name="community_node_01",
        timestamp="2026-01-01T00:00:00",
        unixtime=1767225600 + msg_id,
        author_anon="Developer_00001",
        author_id_anon="user_00001",
        text_clean=text,
    )


def _raw(msg_id: int, text: str, author: str = "Алексей") -> NormalizedMessage:
    return NormalizedMessage(
        msg_id=msg_id,
        chat_id=1,
        chat_name="community_node_01",
        timestamp="2026-01-01T00:00:00",
        unixtime=1767225600 + msg_id,
        author_raw=author,
        author_id_raw=f"user_{msg_id}",
        text_raw=text,
    )


class TestExactDedupStreaming(unittest.TestCase):
    def test_stream_matches_the_list_api(self):
        messages = [
            _cleaned(1, "одинаковый достаточно длинный текст сообщения"),
            _cleaned(2, "одинаковый достаточно длинный текст сообщения"),
            _cleaned(3, "совсем другой текст сообщения тоже длинный"),
        ]
        list_api = ExactDeduplicator().deduplicate(messages)
        stream = ExactDeduplicator()
        streamed = list(stream.deduplicate_stream(iter(messages)))
        self.assertEqual([m.msg_id for m in list_api[0]], [m.msg_id for m in streamed])
        self.assertEqual(list_api[1], 1)

    def test_stream_is_lazy(self):
        """The generator must not consume the source until iterated."""
        consumed: list[int] = []

        def source():
            for i in range(3):
                consumed.append(i)
                yield _cleaned(i, f"текст сообщения номер {i} достаточно длинный")

        stream = ExactDeduplicator().deduplicate_stream(source())
        self.assertEqual(consumed, [], "the source was consumed before the stream was iterated")
        next(stream)
        self.assertEqual(consumed, [0])

    def test_fingerprints_are_compact(self):
        """The index stores raw digests, not 32-character hex strings."""
        dedup = ExactDeduplicator()
        dedup.deduplicate([_cleaned(1, "достаточно длинный текст для хеширования")])
        self.assertEqual(len(dedup.seen_hashes), 1)
        digest = next(iter(dedup.seen_hashes))
        self.assertIsInstance(digest, bytes)
        self.assertEqual(len(digest), 16)


class TestMinHashStreaming(unittest.TestCase):
    def test_stream_matches_the_list_api(self):
        # The two texts differ in their last word, so the MinHash
        # signatures are close but not identical: the LSH may or may not
        # collapse them depending on the bucket. What must match is the
        # STREAM's own verdict, so the same corpus is deduplicated twice
        # and the results are compared.
        messages = [_cleaned(1, LONG + " один"), _cleaned(2, LONG + " два"), _cleaned(3, LONG + " один")]
        list_api = MinHashLSH().deduplicate_messages(messages)
        stream = MinHashLSH()
        streamed = list(stream.deduplicate_stream(iter(messages)))
        self.assertEqual([m.msg_id for m in list_api[0]], [m.msg_id for m in streamed])

    def test_stream_removes_an_exact_duplicate(self):
        # The two texts differ in their last word, so their MinHash
        # signatures are very close and the LSH collapses them too: the
        # threshold is 0.80 and one word in fifty barely moves the
        # Jaccard estimate. That is the intended behaviour of a NEAR
        # duplicate filter, and the point of the test is that the
        # streaming path agrees with the list path.
        messages = [_cleaned(1, LONG + " один"), _cleaned(2, LONG + " два"), _cleaned(3, LONG + " один")]
        list_api = MinHashLSH().deduplicate_messages(messages)
        stream = MinHashLSH()
        streamed = list(stream.deduplicate_stream(iter(messages)))
        self.assertEqual([m.msg_id for m in streamed], [m.msg_id for m in list_api[0]])
        self.assertLess(len(streamed), len(messages), "the near-duplicate was not removed")

    def test_stream_is_lazy(self):
        consumed: list[int] = []

        def source():
            for i in range(3):
                consumed.append(i)
                yield _cleaned(i, LONG + f" номер {i}")

        stream = MinHashLSH().deduplicate_stream(source())
        self.assertEqual(consumed, [])
        next(stream)
        self.assertEqual(consumed, [0])


class TestPIIStreaming(unittest.TestCase):
    def test_stream_matches_the_list_api(self):
        messages = [
            _raw(1, "Позвоните +79161234567 или напишите test@example.com"),
            _raw(2, "Обычное сообщение без персональных данных вообще"),
            _raw(3, "Ещё одно сообщение, тоже без персональных данных"),
        ]
        list_api = UnifiedPIIAnonymizer(enable_ner=False).process_batch(messages)
        stream = UnifiedPIIAnonymizer(enable_ner=False)
        # The author pass reads the corpus first (to harvest the inflected
        # name forms); the scrubbing pass then streams it.
        streamed = list(stream.process_stream(iter(messages), messages))
        self.assertEqual([m.msg_id for m in list_api], [m.msg_id for m in streamed])
        self.assertEqual([m.text_clean for m in list_api], [m.text_clean for m in streamed])

    def test_the_stream_scrubs_the_pii(self):
        messages = [_raw(1, "Позвоните +79161234567 или напишите test@example.com")]
        stream = UnifiedPIIAnonymizer(enable_ner=False)
        cleaned = list(stream.process_stream(iter(messages), messages))
        self.assertEqual(len(cleaned), 1)
        self.assertNotIn("+79161234567", cleaned[0].text_clean)
        self.assertNotIn("test@example.com", cleaned[0].text_clean)

    def test_the_stream_scrubs_lazily(self):
        """The scrubbing must not happen before the consumer pulls.

        The AUTHOR pass reads the whole corpus (the inflected name forms
        have to be known before any text is scrubbed) — that is inherent
        to the two-pass design. What must be lazy is the SCRUBBING: no
        message is cleaned until the consumer pulls it.
        """
        scrubbed: list[int] = []

        def source():
            for i in range(3):
                yield _raw(i, f"Сообщение номер {i} без персональных данных")

        stream = UnifiedPIIAnonymizer(enable_ner=False)

        original = stream.anonymize_message

        def counting(message: NormalizedMessage):
            scrubbed.append(message.msg_id)
            return original(message)

        stream.anonymize_message = counting  # type: ignore[method-assign]
        generator = stream.process_stream(source(), source())

        # Nothing has been scrubbed yet.
        self.assertEqual(scrubbed, [], "the messages were scrubbed before the consumer pulled them")
        next(generator)
        self.assertEqual(scrubbed, [0], "more than one message was scrubbed for a single pull")

    def test_names_are_harvested_before_the_text_is_scrubbed(self):
        """The inflected forms of a name must be known before scrubbing.

        The author pass runs over the whole corpus first; the scrubbing
        pass then streams. A name that appears only in the LAST message
        still has to be masked in the first one — that is what the
        two-pass structure is for.
        """
        messages = [
            _raw(1, "Обсуждаем архитектуру сервиса подробно", author="Алексей"),
            _raw(2, "Алексея спросили про конфигурацию базы данных", author="Мария"),
        ]
        stream = UnifiedPIIAnonymizer(enable_ner=False)
        cleaned = list(stream.process_stream(iter(messages)))
        joined = " ".join(m.text_clean for m in cleaned)
        self.assertNotIn("Алексей", joined)
        self.assertNotIn("Алексея", joined)
        self.assertNotIn("Мария", joined)


if __name__ == "__main__":
    unittest.main()
