"""
demo_corpus.py

A small synthetic document corpus written in formal documentation style,
paired with casual test queries that a real user might type. This
mismatch — casual question vocabulary vs. formal document vocabulary —
is exactly the failure mode HyDE is designed to close.

Domain: a generic support/documentation knowledge base (payments,
account access, notifications, data export, integrations) — deliberately
generic so the demo isn't tied to any specific client or product.
"""

from hyde import Document

DOCUMENTS = [
    Document(
        doc_id="doc_payment_retry",
        text=(
            "Recurring transaction failures may occur due to insufficient "
            "account balance, expired card credentials, or bank-side fraud "
            "flagging. The system automatically schedules a retry attempt "
            "within 24 hours. Refer to the retry policy section for the "
            "maximum number of attempts permitted."
        ),
    ),
    Document(
        doc_id="doc_account_lockout",
        text=(
            "Repeated failed authentication attempts trigger a temporary "
            "account lockout as a security measure. The lockout duration "
            "increases with each subsequent failed attempt. Users may "
            "request an immediate unlock via the verified recovery email "
            "or phone number on file."
        ),
    ),
    Document(
        doc_id="doc_notification_delay",
        text=(
            "Notification delivery latency can result from provider-side "
            "queuing, device connectivity issues, or user-level "
            "notification preferences that suppress non-critical alerts. "
            "Delivery is not guaranteed within a fixed time window under "
            "high system load."
        ),
    ),
    Document(
        doc_id="doc_data_export",
        text=(
            "Bulk data export requests are processed asynchronously and "
            "the resulting file is made available via a time-limited "
            "download link. Export size is capped per the account's "
            "current subscription tier. Large exports may take up to "
            "several hours to complete."
        ),
    ),
    Document(
        doc_id="doc_integration_sync",
        text=(
            "Third-party integration synchronization failures are most "
            "commonly caused by an expired API token, a revoked "
            "permission scope, or a rate limit imposed by the external "
            "provider. Reauthorizing the integration typically resolves "
            "token-related failures."
        ),
    ),
    Document(
        doc_id="doc_refund_window",
        text=(
            "Refund eligibility is determined by the policy in effect at "
            "the time of purchase. Standard refund windows apply unless "
            "superseded by a promotional or regional policy variant. "
            "Refund processing time varies by original payment method."
        ),
    ),
]

# Casual queries a real user might type, paired with the doc_id that
# actually answers them — used to evaluate whether retrieval finds the
# right document despite the register mismatch.
TEST_QUERIES = [
    ("why did my payment fail twice this month", "doc_payment_retry"),
    ("i keep getting locked out of my account", "doc_account_lockout"),
    ("why are my notifications showing up late", "doc_notification_delay"),
    ("how long does it take to download all my data", "doc_data_export"),
    ("my third party app stopped syncing", "doc_integration_sync"),
    ("can i get my money back for this order", "doc_refund_window"),
]
