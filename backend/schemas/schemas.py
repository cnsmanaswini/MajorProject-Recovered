"""
Pydantic schemas for MindGram API
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime


# ── User ────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    display_name: str
    avatar_url: Optional[str] = ""
    bio: Optional[str] = ""


class UserOut(BaseModel):
    id: int
    username: str
    display_name: str
    avatar_url: str
    bio: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Post ────────────────────────────────────────────────────

class PostCreate(BaseModel):
    user_id: int
    content: str
    image_url: Optional[str] = ""
    video_url: Optional[str] = ""
    location: Optional[str] = ""
    is_reel: bool = False


class PostMediaOut(BaseModel):
    id: int
    post_id: int
    media_type: str
    url: str
    public_id: str = ""
    position: int
    created_at: datetime

    class Config:
        from_attributes = True


class PostOut(BaseModel):
    id: int
    user_id: int
    content: str
    image_url: str
    video_url: str
    is_reel: bool
    location: str
    created_at: datetime
    sentiment: str
    sentiment_score: float
    emotion: str
    emotion_score: float
    sarcasm: bool
    sarcasm_score: float
    risk_score: float
    feed_score: float
    topics: List[str] = Field(default_factory=list)
    likes_count: int
    comments_count: int
    media: List[PostMediaOut] = Field(default_factory=list)
    is_liked: bool = False
    is_wellness: bool = False
    author: Optional[UserOut] = None

    @field_validator("topics", mode="before")
    @classmethod
    def coerce_topics(cls, v):
        return v or []

    class Config:
        from_attributes = True


# ── Comment ─────────────────────────────────────────────────

class CommentCreate(BaseModel):
    post_id: int
    user_id: int
    content: str


class CommentOut(BaseModel):
    id: int
    post_id: int
    user_id: int
    content: str
    created_at: datetime
    sentiment: str
    user: Optional[UserOut] = None

    class Config:
        from_attributes = True


# ── Message ─────────────────────────────────────────────────

class MessageCreate(BaseModel):
    sender_id: int
    receiver_id: int
    content: str


class MessageOut(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    content: str
    created_at: datetime
    sentiment: str
    emotion: str
    risk_score: float

    class Config:
        from_attributes = True


# ── Interaction ─────────────────────────────────────────────

class InteractionCreate(BaseModel):
    user_id: int
    post_id: int
    action: Literal["like", "unlike", "not_interested", "share"] = "like"


# ── Analytics ───────────────────────────────────────────────

class EmotionPoint(BaseModel):
    timestamp: datetime
    sentiment_score: float
    emotion: str
    risk_score: float
    agent_action: str


class AnalyticsOut(BaseModel):
    user_id: int
    emotion_timeline: List[EmotionPoint]
    current_risk: float
    dominant_emotion: str
    average_sentiment: float
    intervention_count: int


# ── Agents ──────────────────────────────────────────────────

class AgentStatusOut(BaseModel):
    user_id: int
    risk_level: str
    decision: str
    intervention: str
    rag_suggestion: str
    timestamp: datetime

    class Config:
        from_attributes = True


# ── AI Pipeline Internal ─────────────────────────────────────

class PipelineResult(BaseModel):
    sentiment: str
    sentiment_score: float
    emotion: str
    emotion_score: float
    sarcasm: bool
    sarcasm_score: float
    # Dissociation / emotional-numbness signal (rule-based, independent of
    # the emotion classifier — see detect_numbness_signal() in pipeline.py).
    numbness: bool = False
    numbness_score: float = 0.0
    risk_score: float
    feed_score: float


class ReportCreate(BaseModel):
    user_id: int
    post_id: int
    reason: Literal["spam", "harassment", "self_harm", "other"] = "other"
    details: Optional[str] = None


class ImpressionCreate(BaseModel):
    user_id: int
    post_id: int
    dwell_ms: int


# ── PASTE INTO schemas/schemas.py ──────────────────────────────
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class StoryCreate(BaseModel):
    user_id: int
    image_url: Optional[str] = ""
    video_url: Optional[str] = ""
    text: Optional[str] = ""


class StoryAuthorOut(BaseModel):
    id: int
    username: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True


class StoryOut(BaseModel):
    id: int
    user_id: int
    image_url: str
    video_url: str
    text: str
    created_at: datetime
    expires_at: Optional[datetime]
    views: int
    sentiment: str
    risk_score: float
    author: Optional[StoryAuthorOut] = None
    viewed_by_me: bool = False

    class Config:
        from_attributes = True


class StoryViewCreate(BaseModel):
    viewer_id: int


class StoryViewerOut(BaseModel):
    """One entry in the 'who viewed my story' list."""
    viewer_id: int
    username: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    viewed_at: datetime

    class Config:
        from_attributes = True


class ViewedStoryOut(BaseModel):
    """One entry in 'stories I have viewed' (personal history)."""
    story_id: int
    author_id: int
    author_username: str
    viewed_at: datetime

    class Config:
        from_attributes = True