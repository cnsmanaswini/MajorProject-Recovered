import asyncio
from sqlalchemy import select
from models.database import AsyncSessionLocal
from models.models import AgentDecision
from services.algorithm import build_feed, silent_ai_adjustment

USER_ID = 8

async def check():
    async with AsyncSessionLocal() as db:
        agent_result = await db.execute(
            select(AgentDecision)
            .where(AgentDecision.user_id == USER_ID)
            .order_by(AgentDecision.timestamp.desc())
            .limit(1)
        )
        agent_row = agent_result.scalar_one_or_none()
        risk_map = {"low": 0.1, "moderate": 0.45, "high": 0.70, "critical": 0.92}
        user_risk = risk_map.get(agent_row.risk_level, 0.1) if agent_row else 0.1
        print(f"user_risk={user_risk} (risk_level={agent_row.risk_level if agent_row else None})")
        print()

        posts = await build_feed(USER_ID, db, limit=15)
        for i, p in enumerate(posts, 1):
            adj = silent_ai_adjustment(p, user_risk)
            print(f'{i:2}. emotion={p.emotion:8} risk={p.risk_score:.2f} adjustment={adj:+.4f}  "{p.content[:50]}"')

asyncio.run(check())
