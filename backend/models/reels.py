# backend/models/reels.py
#
# ASSUMPTIONS (adjust imports to match your actual project):
#   - You already have: backend/database.py exposing `Base`
#   - You already have: models/user.py -> User, models/post.py -> Post, PostMedia
#   - Post has: id, user_id, caption, created_at, is_reel (add this column if missing),
#     media_type ("image" | "video" | "carousel")
#   - You already have Like, Comment, Follow models. If your names differ, rename
#     the FKs below to match (e.g. Like -> PostLike).
#
# Run `alembic revision --autogenerate -m "add reels tables"` after dropping this in.

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from models.database import Base  # adjust import path if your Base lives elsewhere


class Audio(Base):
    """A piece of audio/music that can be attached to a reel (Post with is_reel=True)."""
    __tablename__ = "audios"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=True)
    audio_url = Column(String(500), nullable=False)  # Cloudinary URL for the audio track
    duration_seconds = Column(Float, nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # null = "original audio" owner is the post creator
    created_at = Column(DateTime(timezone=True), server_default=func.now())

   # reels = relationship("Post", back_populates="audio")  # TODO: unfinished feature — Post has no `audio` relationship / is_reel column yet


class ReelView(Base):
    """Tracks a single viewing session of a reel by a user — the core signal for
    watch-time-based ranking and for the temporal/behavioral side of your AI layer."""
    __tablename__ = "reel_views"

    id = Column(Integer, primary_key=True, index=True)
    reel_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    watch_seconds = Column(Float, default=0.0)       # how long they watched
    reel_duration = Column(Float, nullable=True)     # total length, for watch % calc
    completed = Column(Boolean, default=False)       # watched to end (or looped once)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SavedReel(Base):
    """A user's saved/bookmarked reel collection."""
    __tablename__ = "saved_reels"
    __table_args__ = (UniqueConstraint("user_id", "reel_id", name="uq_user_reel_save"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    reel_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NotInterested(Base):
    """User marked a reel as 'not interested' — feeds negative signal into ranking
    and permanently excludes it from that user's feed."""
    __tablename__ = "reel_not_interested"
    __table_args__ = (UniqueConstraint("user_id", "reel_id", name="uq_user_reel_not_interested"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    reel_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    reason = Column(String(100), nullable=True)  # optional: "not_relevant", "seen_too_much", etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Report(Base):
    """Content report."""
    __tablename__ = "reel_reports"

    id = Column(Integer, primary_key=True, index=True)
    reporter_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    reel_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    reason = Column(String(100), nullable=False)  # "spam", "harassment", "self_harm", "nudity", "other"
    details = Column(Text, nullable=True)
    status = Column(String(20), default="pending")  # pending | reviewed | actioned
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ShareEvent(Base):
    """Tracks a reel share (internal share to another user, or external link copy)."""
    __tablename__ = "share_events"

    id = Column(Integer, primary_key=True, index=True)
    reel_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    shared_to_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # null if it was a copy-link/external share
    method = Column(String(20), default="link")  # "dm" | "link" | "external"
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# --- Additions expected on your existing Post model (models/post.py) ---
# is_reel = Column(Boolean, default=False, index=True)
# audio_id = Column(Integer, ForeignKey("audios.id"), nullable=True)
# audio = relationship("Audio", back_populates="reels")
#
# If your Post model doesn't already track counts, add (or compute via COUNT queries):
# like_count / comment_count / share_count / save_count / view_count (Integer, default=0)