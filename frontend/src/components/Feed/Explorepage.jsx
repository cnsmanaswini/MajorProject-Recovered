import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import {
  Heart, MessageCircle, Play, Layers, X, ChevronLeft, ChevronRight,
  Bookmark, Search, Flame,
} from "lucide-react";
import { useAuth } from "../../context/AuthContext";

// ---------------------------------------------------------------------------
// This page is now wired to the real backend:
//   GET /api/feed/explore  → real posts (public accounts, last 14 days)
//   GET /api/feed/trending → real trending topics
// Captions come from post.content. No mock data, no fabricated engagement.
// ---------------------------------------------------------------------------

const PAGE_SIZE = 30;
const WIDE_POSITIONS = new Set([3, 10]);
const FALLBACK_AVATAR = (seed) => `https://i.pravatar.cc/150?u=${seed}`;

function formatCount(n) {
  const v = n ?? 0;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return String(v);
}

// Same pattern FeedPage.jsx uses: prefer the media[] array, fall back to
// the legacy single image_url/video_url columns.
function getPostMedia(post) {
  if (post.media?.length) {
    return [...post.media].sort((a, b) => (a.position || 0) - (b.position || 0));
  }
  if (post.image_url) return [{ id: "legacy-image", media_type: "image", url: post.image_url }];
  if (post.video_url) return [{ id: "legacy-video", media_type: "video", url: post.video_url }];
  return [];
}

// post.topics is real (hashtags + soft tags, set server-side). Fall back to
// pulling #hashtags out of the caption itself if topics is empty.
function getHashtags(post) {
  if (post.topics?.length) {
    return post.topics.map((t) => (t.startsWith("#") ? t : `#${t}`));
  }
  return (post.content || "").match(/#\w+/g) || [];
}

// post.emotion is a real field from the backend pipeline (CLIP for media,
// text classifier for captions) — this just maps it to something visible.
// Not every possible label is guaranteed here; unknown ones fall back to a
// plain dot so the badge never looks broken.
const EMOTION_EMOJI = {
  joy: "😄", excitement: "🤩", pride: "🥲", calm: "😌", love: "🥰",
  surprise: "😮", neutral: "😐", sadness: "😔", tiredness: "🥱",
  frustration: "😤", stress: "😩", anger: "😠", fear: "😨", disgust: "😖",
};

function EmotionBadge({ emotion, sarcasm }) {
  if (!emotion) return null;
  const emoji = EMOTION_EMOJI[emotion.toLowerCase()] || "•";
  return (
    <span className="inline-flex items-center gap-2">
      <span className="text-xs bg-neutral-800 text-neutral-200 px-2 py-0.5 rounded-full capitalize">
        {emoji} {emotion}
      </span>
      {sarcasm && (
        <span className="text-xs bg-neutral-800 text-amber-300 px-2 py-0.5 rounded-full">
          😏 sarcastic
        </span>
      )}
    </span>
  );
}

// Reliable video-thumbnail rendering: the `#t=0.5` URL-fragment trick only
// half-works across browsers (Chrome sometimes honors it, Firefox mostly
// doesn't without playback). This instead seeks the actual <video> element
// once its metadata is ready, which every browser paints correctly.
function VideoThumbnail({ src, className }) {
  const videoRef = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const seekToFrame = () => {
      try {
        v.currentTime = Math.min(0.5, (v.duration || 1) / 4);
      } catch {
        // duration not available yet on some browsers; onloadeddata below covers it
      }
    };
    const onSeeked = () => setReady(true);
    v.addEventListener("loadedmetadata", seekToFrame);
    v.addEventListener("loadeddata", seekToFrame);
    v.addEventListener("seeked", onSeeked);
    return () => {
      v.removeEventListener("loadedmetadata", seekToFrame);
      v.removeEventListener("loadeddata", seekToFrame);
      v.removeEventListener("seeked", onSeeked);
    };
  }, [src]);

  return (
    <video
      ref={videoRef}
      src={src}
      muted
      playsInline
      preload="auto"
      className={className}
      style={{ opacity: ready ? 1 : 0, transition: "opacity 150ms ease", backgroundColor: "#171717" }}
    />
  );
}

// ---------------------------------------------------------------------------
// UI pieces
// ---------------------------------------------------------------------------

function SearchBar({ value, onChange }) {
  return (
    <div className="relative px-1 pt-1">
      <div className="flex items-center gap-2 bg-neutral-900 rounded-lg px-3 py-2">
        <Search className="w-4 h-4 text-neutral-400 shrink-0" />
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Search captions, usernames, hashtags"
          className="bg-transparent outline-none text-sm text-white placeholder-neutral-500 w-full"
        />
        {value && (
          <button onClick={() => onChange("")} aria-label="Clear search" className="text-neutral-500 hover:text-neutral-300">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}

function CategoryChips({ chips, active, onSelect }) {
  if (chips.length <= 1) return null; // nothing but "For You" to show yet
  return (
    <div className="flex gap-2 px-1 py-3 overflow-x-auto no-scrollbar">
      {chips.map((c) => (
        <button
          key={c}
          onClick={() => onSelect(c)}
          className={`shrink-0 px-3 py-1.5 rounded-full text-sm border transition-colors ${
            active === c
              ? "bg-white text-black border-white"
              : "bg-transparent text-neutral-300 border-neutral-700 hover:border-neutral-500"
          }`}
        >
          {c === "for_you" ? "For You" : c}
        </button>
      ))}
    </div>
  );
}

function TrendingHashtagsRow({ topics }) {
  if (!topics.length) return null;
  return (
    <div className="px-2 pb-3 flex items-center gap-2 flex-wrap">
      <span className="flex items-center gap-1 text-xs text-orange-400 font-medium">
        <Flame className="w-3.5 h-3.5" /> Trending
      </span>
      {topics.map((t) => (
        <span key={t.topic} className="text-xs text-neutral-300 bg-neutral-900 px-2 py-1 rounded-full">
          {t.topic}
        </span>
      ))}
    </div>
  );
}

function ExploreTile({ post, wide, onOpen }) {
  const [hovered, setHovered] = useState(false);
  const media = getPostMedia(post)[0];
  if (!media) return null; // skip text-only posts in the grid
  const isVideo = media.media_type === "video" || post.is_reel;

  return (
    <button
      type="button"
      onClick={() => onOpen(post)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`relative overflow-hidden bg-neutral-900 ${wide ? "col-span-2 row-span-2" : ""}`}
      style={{ aspectRatio: "1 / 1" }}
    >
      {isVideo ? (
        <VideoThumbnail
          src={media.url}
          className="w-full h-full object-cover"
        />
      ) : (
        <img
          src={media.url}
          alt={post.content ? post.content.slice(0, 80) : ""}
          loading="lazy"
          className="w-full h-full object-cover"
          style={{ transform: hovered ? "scale(1.03)" : "scale(1)", transition: "transform 200ms ease" }}
        />
      )}

      {isVideo && (
        <div className="absolute top-2 right-2 text-white" style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.5))" }}>
          <Play className="w-4 h-4 fill-white" />
        </div>
      )}
      {getPostMedia(post).length > 1 && (
        <div className="absolute top-2 left-2 text-white" style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.5))" }}>
          <Layers className="w-4 h-4 fill-white" />
        </div>
      )}

      {post.emotion && (
        <span
          className="absolute bottom-1.5 left-1.5 text-[11px]"
          style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.6))" }}
        >
          {EMOTION_EMOJI[post.emotion.toLowerCase()] || "•"}
        </span>
      )}

      <div
        className="absolute inset-0 flex items-center justify-center gap-6 text-white font-semibold"
        style={{ background: "rgba(0,0,0,0.3)", opacity: hovered ? 1 : 0, transition: "opacity 120ms ease" }}
      >
        <span className="flex items-center gap-1.5 text-sm">
          <Heart className="w-4 h-4 fill-white" />
          {formatCount(post.likes_count)}
        </span>
        <span className="flex items-center gap-1.5 text-sm">
          <MessageCircle className="w-4 h-4 fill-white" />
          {formatCount(post.comments_count)}
        </span>
      </div>
    </button>
  );
}

function ReelsRow({ posts, onOpen }) {
  const reels = posts.filter((p) => p.is_reel && getPostMedia(p).length).slice(0, 10);
  if (!reels.length) return null;
  return (
    <div className="px-1 pb-3">
      <div className="flex items-center gap-1.5 px-1 pb-2 text-sm text-neutral-300">
        <Play className="w-3.5 h-3.5 fill-neutral-300" />
        <span>Reels</span>
      </div>
      <div className="flex gap-2 overflow-x-auto no-scrollbar px-1">
        {reels.map((r) => {
          const media = getPostMedia(r)[0];
          return (
            <button
              key={r.id}
              onClick={() => onOpen(r)}
              className="relative shrink-0 rounded-lg overflow-hidden bg-neutral-900"
              style={{ width: 110, aspectRatio: "9 / 16" }}
            >
              <VideoThumbnail src={media.url} className="w-full h-full object-cover" />
              <div className="absolute bottom-1.5 left-1.5 flex items-center gap-1 text-white text-[11px]" style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.6))" }}>
                <Play className="w-3 h-3 fill-white" />
                {formatCount(r.likes_count)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Lightbox({ posts, index, onClose, onNavigate }) {
  const post = posts[index];
  const media = getPostMedia(post)[0];
  const author = post.author || {};
  const isVideo = media?.media_type === "video" || post.is_reel;

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") onNavigate(-1);
      if (e.key === "ArrowRight") onNavigate(1);
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose, onNavigate]);

  const hasPrev = index > 0;
  const hasNext = index < posts.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.92)" }} onClick={onClose}>
      <button onClick={onClose} className="absolute top-4 right-4 text-white/80 hover:text-white p-2" aria-label="Close">
        <X className="w-6 h-6" />
      </button>

      {hasPrev && (
        <button onClick={(e) => { e.stopPropagation(); onNavigate(-1); }} className="absolute left-2 sm:left-6 text-white/70 hover:text-white p-2" aria-label="Previous">
          <ChevronLeft className="w-8 h-8" />
        </button>
      )}
      {hasNext && (
        <button onClick={(e) => { e.stopPropagation(); onNavigate(1); }} className="absolute right-2 sm:right-6 text-white/70 hover:text-white p-2" aria-label="Next">
          <ChevronRight className="w-8 h-8" />
        </button>
      )}

      <div className="bg-black w-full max-w-4xl mx-4 max-h-[88vh] flex flex-col sm:flex-row overflow-hidden rounded-sm border border-neutral-800" onClick={(e) => e.stopPropagation()}>
        <div className="bg-black flex items-center justify-center sm:w-[62%] max-h-[50vh] sm:max-h-[88vh]">
          {isVideo ? (
            <video src={media.url} controls className="max-h-[50vh] sm:max-h-[88vh] w-full object-contain" />
          ) : (
            <img src={media?.url} alt="" className="max-h-[50vh] sm:max-h-[88vh] w-full object-contain" />
          )}
        </div>

        <div className="sm:w-[38%] flex flex-col text-white">
          <div className="flex items-center gap-3 px-4 py-3 border-b border-neutral-800">
            <img src={author.avatar_url || FALLBACK_AVATAR(author.username || post.user_id)} alt="" className="w-8 h-8 rounded-full object-cover" />
            <span className="font-semibold text-sm">{author.username || "unknown"}</span>
            <span className="ml-auto">
              <EmotionBadge emotion={post.emotion} sarcasm={post.sarcasm} />
            </span>
          </div>

          <div className="flex-1 px-4 py-3 overflow-y-auto">
            <div className="flex gap-3 text-sm">
              <img src={author.avatar_url || FALLBACK_AVATAR(author.username || post.user_id)} alt="" className="w-8 h-8 rounded-full object-cover shrink-0" />
              <p>
                <span className="font-semibold mr-1.5">{author.username || "unknown"}</span>
                {post.content || <span className="text-neutral-500">No caption</span>}
              </p>
            </div>
          </div>

          <div className="px-4 pt-3 pb-2 border-t border-neutral-800">
            <div className="flex items-center gap-4 mb-2">
              <Heart className="w-6 h-6" />
              <MessageCircle className="w-6 h-6" />
              <Bookmark className="w-6 h-6 ml-auto" />
            </div>
            <p className="text-sm font-semibold">{formatCount(post.likes_count)} likes</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ExplorePage() {
  const { api } = useAuth();

  const [allPosts, setAllPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState(null);
  const [lightboxIndex, setLightboxIndex] = useState(null);
  const [activeTopic, setActiveTopic] = useState("for_you");
  const [query, setQuery] = useState("");
  const [trending, setTrending] = useState([]);

  const sentinelRef = useRef(null);
  const offsetRef = useRef(0);
  const loadingMoreRef = useRef(false);

  const fetchPage = useCallback(async (offset) => {
    const res = await api.get("/feed/explore", { params: { limit: PAGE_SIZE, offset } });
    return res.data;
  }, [api]);

  // Initial load
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [posts, trendingRes] = await Promise.all([
          fetchPage(0),
          api.get("/feed/trending", { params: { limit: 8 } }).catch(() => ({ data: [] })),
        ]);
        if (cancelled) return;
        offsetRef.current = posts.length;
        setAllPosts(posts);
        setTrending(trendingRes.data || []);
        setHasMore(posts.length === PAGE_SIZE);
      } catch (err) {
        if (!cancelled) setError("Couldn't load explore right now. Pull to refresh in a bit.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [fetchPage, api]);

  const loadMore = useCallback(async () => {
    if (loadingMoreRef.current || !hasMore) return;
    loadingMoreRef.current = true;
    setLoadingMore(true);
    try {
      const next = await fetchPage(offsetRef.current);
      offsetRef.current += next.length;
      setAllPosts((prev) => {
        const seen = new Set(prev.map((p) => p.id));
        return [...prev, ...next.filter((p) => !seen.has(p.id))];
      });
      setHasMore(next.length === PAGE_SIZE);
    } catch {
      // leave hasMore as-is; the sentinel will just retry on next intersect
    } finally {
      loadingMoreRef.current = false;
      setLoadingMore(false);
    }
  }, [fetchPage, hasMore]);

  useEffect(() => {
    const node = sentinelRef.current;
    if (!node || loading) return;
    const observer = new IntersectionObserver(
      (entries) => entries[0].isIntersecting && loadMore(),
      { rootMargin: "600px 0px" }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [loading, loadMore]);

  // Real category chips, built from hashtags actually present in loaded posts.
  const topicChips = useMemo(() => {
    const counts = {};
    allPosts.forEach((p) => getHashtags(p).forEach((h) => { counts[h] = (counts[h] || 0) + 1; }));
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 7).map(([h]) => h);
    return ["for_you", ...top];
  }, [allPosts]);

  const topicFiltered = useMemo(() => {
    if (activeTopic === "for_you") return allPosts;
    return allPosts.filter((p) => getHashtags(p).includes(activeTopic));
  }, [allPosts, activeTopic]);

  const displayedPosts = useMemo(() => {
    if (!query.trim()) return topicFiltered;
    const q = query.trim().toLowerCase();
    return topicFiltered.filter(
      (p) =>
        (p.author?.username || "").toLowerCase().includes(q) ||
        (p.content || "").toLowerCase().includes(q) ||
        getHashtags(p).some((h) => h.toLowerCase().includes(q))
    );
  }, [topicFiltered, query]);

  const openAt = (post) => {
    const idx = displayedPosts.findIndex((p) => p.id === post.id);
    if (idx !== -1) setLightboxIndex(idx);
  };

  const navigate = (delta) => {
    setLightboxIndex((prev) => {
      if (prev === null) return prev;
      const next = prev + delta;
      if (next < 0 || next >= displayedPosts.length) return prev;
      if (next >= displayedPosts.length - 3) loadMore();
      return next;
    });
  };

  return (
    <div className="min-h-screen" style={{ background: "#000" }}>
      <style>{`.no-scrollbar::-webkit-scrollbar{display:none}.no-scrollbar{-ms-overflow-style:none;scrollbar-width:none}`}</style>
      <div className="max-w-4xl mx-auto px-1 py-3">
        <SearchBar value={query} onChange={setQuery} />
        <CategoryChips chips={topicChips} active={activeTopic} onSelect={setActiveTopic} />

        {!loading && !query && (
          <>
            <TrendingHashtagsRow topics={trending} />
            <ReelsRow posts={topicFiltered} onOpen={openAt} />
          </>
        )}

        {error && (
          <div className="text-center text-red-400 text-sm py-8">{error}</div>
        )}

        {loading ? (
          <div className="grid grid-cols-3 gap-1" style={{ gridAutoFlow: "dense" }}>
            {Array.from({ length: PAGE_SIZE }).map((_, i) => (
              <div
                key={i}
                className={`bg-neutral-900 animate-pulse ${WIDE_POSITIONS.has(i % 15) ? "col-span-2 row-span-2" : ""}`}
                style={{ aspectRatio: "1 / 1" }}
              />
            ))}
          </div>
        ) : displayedPosts.length === 0 ? (
          <div className="text-center text-neutral-500 text-sm py-16">
            {query ? `No results for "${query}"` : "No posts to explore yet."}
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-1" style={{ gridAutoFlow: "dense" }}>
            {displayedPosts.map((post, idx) => (
              <ExploreTile key={post.id} post={post} wide={WIDE_POSITIONS.has(idx % 15)} onOpen={openAt} />
            ))}
          </div>
        )}

        {loadingMore && (
          <div className="text-center text-neutral-500 text-xs py-4">Loading more…</div>
        )}
        <div ref={sentinelRef} className="h-1" />
      </div>

      {lightboxIndex !== null && (
        <Lightbox posts={displayedPosts} index={lightboxIndex} onClose={() => setLightboxIndex(null)} onNavigate={navigate} />
      )}
    </div>
  );
}