"""Call record model.

This module defines the database model for storing call metadata.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hermes.models.base import Base

if TYPE_CHECKING:
    from hermes.models.conversation import Conversation


class CallRecord(Base):
    """Call record model.

    Represents a voice call with its metadata and related conversations.
    """

    __tablename__ = "call_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_sid: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    stream_sid: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    account_sid: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # Call status
    status: Mapped[str] = mapped_column(
        String(50),
        default="in_progress",
        nullable=False,
    )  # "in_progress", "completed", "failed", "transferred"

    # Quality metrics
    stt_latency_avg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    llm_latency_avg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    tts_latency_avg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationships
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation",
        back_populates="call",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<CallRecord(id={self.id}, call_sid={self.call_sid}, status={self.status})>"

    def calculate_duration(self) -> float:
        """Calculate call duration in seconds.

        Returns:
            Duration in seconds, or 0 if call hasn't ended.
        """
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return 0.0
