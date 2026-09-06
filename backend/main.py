"""
MindGram — Mental Health-Aware Social Media Platform
FastAPI Backend Entry Point
"""

from dotenv import load_dotenv
load_dotenv()
from routers import reels

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from routers import chat

from config import settings
from models.database import create_tables
from services.cloudinary_service import UPLOAD_ROOT
from routers import auth, users, posts, feed, messages, notifications, analytics, interactions, agents, stories

from ai.pipeline.loader import preload_models
from ai.rag.index import build_rag_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindgram")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & Shutdown"""

    logger.info("🚀 MindGram starting...")

    await create_tables()
    logger.info("✅ Database tables created")

    try:
        preload_models()
        logger.info("✅ AI models loaded")
    except Exception as e:
        logger.warning(f"AI models failed: {e}")

    try:
        build_rag_index()
        logger.info("✅ RAG Index created")
    except Exception as e:
        logger.warning(f"RAG Index failed: {e}")

    yield

    logger.info("🛑 MindGram shutting down")


app = FastAPI(
    title="MindGram API",
    description="Mental Health-Aware Social Media Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(reels.router)
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
# -------------------- CORS --------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router,             prefix="/api/auth",         tags=["Auth"])
app.include_router(users.router,            prefix="/api/users",        tags=["Users"])
app.include_router(posts.router,            prefix="/api/posts",        tags=["Posts"])
app.include_router(feed.router,             prefix="/api/feed",         tags=["Feed"])
app.include_router(messages.router,         prefix="/api/messages",     tags=["Messages"])
app.include_router(notifications.router,    prefix="/api/notifications", tags=["Notifications"])
app.include_router(analytics.router,        prefix="/api/analytics",    tags=["Analytics"])
app.include_router(interactions.router,     prefix="/api/interactions",  tags=["Interactions"])
app.include_router(agents.router,           prefix="/api/agents",       tags=["Agents"])
app.include_router(stories.router,          prefix="/api/stories",      tags=["Stories"])

# Serve locally uploaded media (used when Cloudinary is not configured)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": "2.0.0",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}