"""Conversation history model.

This module defines the database model for storing conversation history.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hermes.models.base import Base

if TYPE_CHECKING:
    from hermes.models.call_record import CallRecord


class Conversation(Base):
    """Conversation turn model.

    Represents a single turn in a conversation between user and assistant.
    """

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("call_records.id"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )  # "user" or "assistant"
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    metadata_json: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )  # JSON string for additional metadata

    # Relationship
    call: Mapped["CallRecord"] = relationship("CallRecord", back_populates="conversations")

    def __repr__(self) -> str:
        """String representation."""
        return f"<Conversation(id={self.id}, role={self.role}, timestamp={self.timestamp})>"
