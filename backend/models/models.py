"""
MindGram Database Models
Full Instagram-like social media + AI mental health pipeline
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(50), unique=True, index=True, nullable=False)
    username_changed_at = Column(DateTime, nullable=True)  # tracks 15-day cooldown
    email           = Column(String(255), unique=True, index=True, nullable=True)
    display_name    = Column(String(100))
    avatar_url      = Column(String(500), default="")
    bio             = Column(Text, default="")
    hashed_password = Column(String(255), nullable=True)  # null for Google OAuth users
    google_id       = Column(String(255), unique=True, nullable=True)
    is_verified     = Column(Boolean, default=False)
    is_private      = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Counts (cached for performance)
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    posts_count     = Column(Integer, default=0)

    # Relationships
    posts           = relationship("Post", back_populates="author", cascade="all, delete")
    stories         = relationship("Story", back_populates="author", cascade="all, delete")
    emotions        = relationship("EmotionLog", back_populates="user", cascade="all, delete")
    messages_sent   = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    notifications   = relationship("Notification", foreign_keys="Notification.user_id", back_populates="user")

    # Follows
    following       = relationship("Follow", foreign_keys="Follow.follower_id", back_populates="follower")
    followers       = relationship("Follow", foreign_keys="Follow.following_id", back_populates="following")


class Follow(Base):
    __tablename__ = "follows"

    id = Column(Integer, primary_key=True, index=True)

    # User who sends the follow request
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # User who receives the follow request
    following_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # requested -> waiting for approval
    # accepted -> following
    status = Column(String(20), default="accepted", nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    follower = relationship(
        "User",
        foreign_keys=[follower_id],
        back_populates="following"
    )

    following = relationship(
        "User",
        foreign_keys=[following_id],
        back_populates="followers"
    )
    
class Post(Base):
    __tablename__ = "posts"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    content     = Column(Text, default="")
    image_url   = Column(String(500), default="")
    video_url   = Column(String(500), default="")
    image_public_id = Column(String(255), default="")   # ← new: needed to delete from Cloudinary
    video_public_id = Column(String(255), default="")   # ← new
    is_reel     = Column(Boolean, default=False)
    location    = Column(String(255), default="")
    created_at  = Column(DateTime, default=datetime.utcnow)

    # AI annotations
    sentiment        = Column(String(20), default="neutral")
    sentiment_score  = Column(Float, default=0.0)
    emotion          = Column(String(30), default="neutral")
    emotion_score    = Column(Float, default=0.0)
    sarcasm          = Column(Boolean, default=False)
    sarcasm_score    = Column(Float, default=0.0)
    risk_score       = Column(Float, default=0.0)
    feed_score       = Column(Float, default=0.5)
    topics           = Column(JSON, default=list)   # hashtags + soft tags for trending
    is_wellness      = Column(Boolean, default=False)  # explicit wellness content flag

    # Engagement
    likes_count    = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    shares_count   = Column(Integer, default=0)
    views_count    = Column(Integer, default=0)

    # Relationships
    author    = relationship("User", back_populates="posts")
    media     = relationship("PostMedia", back_populates="post", cascade="all, delete-orphan", order_by="PostMedia.position")
    comments  = relationship("Comment", back_populates="post", cascade="all, delete")
    likes     = relationship("Like", back_populates="post", cascade="all, delete")


class PostMedia(Base):
    __tablename__ = "post_media"
    id          = Column(Integer, primary_key=True, index=True)
    post_id     = Column(Integer, ForeignKey("posts.id"), nullable=False, index=True)
    media_type  = Column(String(20), nullable=False)  # image | video
    url         = Column(String(500), nullable=False)
    public_id   = Column(String(255), default="")
    position    = Column(Integer, default=0)
    created_at  = Column(DateTime, default=datetime.utcnow)

    post = relationship("Post", back_populates="media")


class Story(Base):
    __tablename__ = "stories"
    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_url        = Column(String(500), default="")
    video_url        = Column(String(500), default="")
    image_public_id  = Column(String(255), default="")
    video_public_id  = Column(String(255), default="")
    text             = Column(Text, default="")
    created_at       = Column(DateTime, default=datetime.utcnow)
    expires_at       = Column(DateTime)  # 24 hours after creation
    views            = Column(Integer, default=0)

    # AI
    sentiment  = Column(String(20), default="neutral")
    risk_score = Column(Float, default=0.0)

    author = relationship("User", back_populates="stories")
    story_views = relationship("StoryView", back_populates="story", cascade="all, delete-orphan")


class StoryView(Base):
    """Individual story view records — powers both 'who viewed my story'
    and 'what stories have I viewed' (personal viewing history).
    One row per (viewer, story) pair — rewatching a story updates
    viewed_at rather than creating duplicates, matching Instagram's
    behavior where views_count means unique viewers, not total opens."""
    __tablename__ = "story_views"
    id         = Column(Integer, primary_key=True, index=True)
    story_id   = Column(Integer, ForeignKey("stories.id"), nullable=False, index=True)
    viewer_id  = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    viewed_at  = Column(DateTime, default=datetime.utcnow)

    story  = relationship("Story", back_populates="story_views")
    viewer = relationship("User")


class Comment(Base):
    __tablename__ = "comments"
    id         = Column(Integer, primary_key=True, index=True)
    post_id    = Column(Integer, ForeignKey("posts.id"), nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    likes      = Column(Integer, default=0)

    # AI annotations — mirrors Post's pipeline output so comments get the
    # same silent risk/emotion tracking, not just a bare sentiment label.
    sentiment        = Column(String(20), default="neutral")
    sentiment_score  = Column(Float, default=0.0)
    emotion          = Column(String(30), default="neutral")
    emotion_score    = Column(Float, default=0.0)
    sarcasm          = Column(Boolean, default=False)
    sarcasm_score    = Column(Float, default=0.0)
    risk_score       = Column(Float, default=0.0)
    feed_score       = Column(Float, default=0.5)

    post = relationship("Post", back_populates="comments")
    user = relationship("User")
    @property
    def author(self):
        return self.user


class Like(Base):
    __tablename__ = "likes"
    id      = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    post = relationship("Post", back_populates="likes")
    user = relationship("User")
    @property
    def author(self):
        return self.user


class Message(Base):
    __tablename__ = "messages"
    id          = Column(Integer, primary_key=True, index=True)
    sender_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    receiver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content     = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)
    is_read     = Column(Boolean, default=False)

    # AI
    sentiment   = Column(String(20), default="neutral")
    emotion     = Column(String(30), default="neutral")
    risk_score  = Column(Float, default=0.0)

    sender   = relationship("User", foreign_keys=[sender_id], back_populates="messages_sent")
    receiver = relationship("User", foreign_keys=[receiver_id])


class EmotionLog(Base):
    __tablename__ = "emotion_logs"
    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    sentiment_score = Column(Float, default=0.0)
    emotion         = Column(String(30), default="neutral")
    emotion_score   = Column(Float, default=0.0)
    risk_score      = Column(Float, default=0.0)
    source          = Column(String(20), default="post")
    agent_action    = Column(String(50), default="none")

    user = relationship("User", back_populates="emotions")


class AgentDecision(Base):
    __tablename__ = "agent_decisions"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp      = Column(DateTime, default=datetime.utcnow)
    risk_level     = Column(String(20), default="low")
    decision       = Column(String(50), default="monitor")
    intervention   = Column(String(100), default="")
    rag_suggestion = Column(Text, default="")
    metadata_json  = Column(JSON, default={})


class Notification(Base):
    __tablename__ = "notifications"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    from_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    type       = Column(String(50))  # like, comment, follow, mention
    message    = Column(Text)
    post_id    = Column(Integer, ForeignKey("posts.id"), nullable=True)
    is_read    = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user      = relationship("User", foreign_keys=[user_id], back_populates="notifications")
    from_user = relationship("User", foreign_keys=[from_user_id])


class UserInterest(Base):
    """Tracks what content a user engages with — powers recommendations."""
    __tablename__ = "user_interests"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    emotion    = Column(String(30))   # what emotions they engage with
    score      = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Report(Base):
    """User-submitted reports on posts — feeds moderation queue and,
    for self-harm reports, the author's risk pipeline."""
    __tablename__ = "reports"
    id          = Column(Integer, primary_key=True, index=True)
    post_id     = Column(Integer, ForeignKey("posts.id"), nullable=False)
    reporter_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reason      = Column(String(50), default="other")   # spam, harassment, self_harm, other
    details     = Column(Text, default="")
    created_at  = Column(DateTime, default=datetime.utcnow)
    reviewed    = Column(Boolean, default=False)

    post     = relationship("Post")
    reporter = relationship("User")


class ImpressionLog(Base):
    """Dwell-time based impression tracking — powers skip-rate inference.
    `skipped` is derived server-side from dwell_ms, not client-trusted."""
    __tablename__ = "impression_logs"
    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    post_id       = Column(Integer, ForeignKey("posts.id"), nullable=False)
    emotion       = Column(String(30))
    dwell_ms      = Column(Integer, default=0)
    skipped       = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)


class NotInterested(Base):
    """Explicit negative feedback — hides post and suppresses author in feed."""
    __tablename__ = "not_interested"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    post_id    = Column(Integer, ForeignKey("posts.id"), nullable=False)
    author_id  = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    post   = relationship("Post")
    user   = relationship("User", foreign_keys=[user_id])
    author = relationship("User", foreign_keys=[author_id])


class Reel(Base):
    __tablename__ = "reels"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, nullable=False)

    caption = Column(String)

    video_url = Column(String, nullable=False)

    likes = Column(Integer, default=0)