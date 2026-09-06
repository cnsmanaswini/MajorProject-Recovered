import React, { useState, useEffect, useRef } from "react";
import {
  Heart,
  MessageCircle,
  Share2,
  Volume2,
  VolumeX,
  Play,
} from "lucide-react";
import { EmotionBadge } from "../Common/Badges.jsx";
import clsx from "clsx";
import { useAuth } from "../../context/AuthContext.jsx";

function reelVideoUrl(reel) {
  if (reel?.video_url) return reel.video_url;
  const media = Array.isArray(reel?.media) ? reel.media : [];
  const video = media.find((m) => m.media_type === "video" && m.url);
  return video?.url || media[0]?.url || "";
}

function ReelCard({ reel, isActive }) {
  const videoRef = useRef(null);

  const [liked, setLiked] = useState(reel.is_liked || false);
  const [likeCount, setLikeCount] = useState(reel.likes_count || 0);
  const [playing, setPlaying] = useState(isActive);
  const [muted, setMuted] = useState(true);

  useEffect(() => {
    setPlaying(isActive);
  }, [isActive]);

  useEffect(() => {
    if (!videoRef.current) return;

    if (playing) {
      videoRef.current.play().catch(() => {});
    } else {
      videoRef.current.pause();
    }
  }, [playing]);

  const handleLike = async () => {
    const token = localStorage.getItem("token");

    try {
      await fetch("/api/interactions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          post_id: reel.id,
          action: liked ? "unlike" : "like",
        }),
      });

      setLiked((prev) => !prev);
      setLikeCount((prev) => (liked ? prev - 1 : prev + 1));
    } catch (err) {
      console.error(err);
    }
  };

  const fmtNum = (n) =>
    n >= 1000 ? `${(n / 1000).toFixed(1)}k` : n;

  return (
    <div className="relative h-[calc(100vh-120px)] min-h-[500px] rounded-2xl overflow-hidden bg-black flex-shrink-0 w-full max-w-sm mx-auto snap-start">

      {reelVideoUrl(reel) ? (
        <video
          ref={videoRef}
          src={reelVideoUrl(reel)}
          muted={muted}
          loop
          playsInline
          preload="metadata"
          className="absolute inset-0 w-full h-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 bg-gray-900 flex items-center justify-center text-white">
          No Video
        </div>
      )}

      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/10 to-black/20" />

      {!playing && (
        <button
          onClick={() => setPlaying(true)}
          className="absolute inset-0 flex items-center justify-center"
        >
          <div className="w-16 h-16 rounded-full bg-black/40 flex items-center justify-center">
            <Play className="text-white ml-1" size={28} />
          </div>
        </button>
      )}

      <button
        onClick={() => setMuted((m) => !m)}
        className="absolute top-4 right-4 w-8 h-8 rounded-full bg-black/40 flex items-center justify-center"
      >
        {muted ? (
          <VolumeX size={16} className="text-white" />
        ) : (
          <Volume2 size={16} className="text-white" />
        )}
      </button>

      <div className="absolute top-14 right-3">
        <EmotionBadge emotion={reel.emotion} />
      </div>

      <div className="absolute right-3 bottom-24 flex flex-col gap-4 items-center">

        <button
          onClick={handleLike}
          className="flex flex-col items-center"
        >
          <div
            className={clsx(
              "w-10 h-10 rounded-full bg-black/40 flex items-center justify-center",
              liked ? "text-red-400" : "text-white"
            )}
          >
            <Heart
              size={20}
              fill={liked ? "currentColor" : "none"}
            />
          </div>

          <span className="text-white text-xs">
            {fmtNum(likeCount)}
          </span>
        </button>

        <button className="flex flex-col items-center">
          <div className="w-10 h-10 rounded-full bg-black/40 flex items-center justify-center text-white">
            <MessageCircle size={20} />
          </div>

          <span className="text-white text-xs">
            {fmtNum(reel.comments_count)}
          </span>
        </button>

        <button className="flex flex-col items-center">
          <div className="w-10 h-10 rounded-full bg-black/40 flex items-center justify-center text-white">
            <Share2 size={20} />
          </div>

          <span className="text-white text-xs">
            Share
          </span>
        </button>

      </div>

      <div className="absolute bottom-0 left-0 right-0 p-4">

        <div className="flex items-center gap-2 mb-2">

          <img
            src={reel.author?.avatar_url || reel.creator?.profile_picture_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${reel.author?.username || reel.creator?.username || "user"}`}
            alt=""
            className="w-8 h-8 rounded-full border border-white/30"
          />

          <span className="text-white font-medium text-sm">
            @{reel.author?.username || reel.creator?.username || "user"}
          </span>

          <button className="pill border border-white/30 text-white text-xs px-2 py-0.5">
            Follow
          </button>

        </div>

        <p className="text-white text-sm leading-relaxed">
          {reel.content || reel.caption || ""}
        </p>

      </div>

    </div>
  );
}
export default function ReelsPage() {
  const { api } = useAuth();
  const [activeIndex, setActiveIndex] = useState(0);
  const [reels, setReels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchReels();
  }, []);

  const fetchReels = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await api.get("/feed/reels");
      const data = response.data;
      const list = Array.isArray(data) ? data : (data?.reels || []);
      setReels(list);
    } catch (err) {
      console.error("Error loading reels:", err);
      setError(err.response?.data?.detail || err.message || "Failed to fetch reels");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center text-white">
        Loading Reels...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center text-red-400">
        {error}
      </div>
    );
  }

  if (!reels.length) {
    return (
      <div className="flex h-screen flex-col items-center justify-center text-white gap-3">
        <h2 className="text-xl font-semibold">
          No reels found
        </h2>

        <button
          onClick={fetchReels}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700"
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-sm mx-auto px-4 py-6">

      <div className="flex items-center justify-between mb-6">

        <h1 className="font-display text-2xl text-white">
          Reels
        </h1>

        <span className="text-xs text-gray-500 font-mono">
          AI-filtered · wellness-safe
        </span>

      </div>

      <div
        className="space-y-4 overflow-y-auto snap-y snap-mandatory"
        style={{
          maxHeight: "calc(100vh - 140px)",
        }}
      >
        {reels.map((reel, index) => (
          <div
            key={reel.id}
            onClick={() => setActiveIndex(index)}
          >
            <ReelCard
              reel={reel}
              isActive={activeIndex === index}
            />
          </div>
        ))}
      </div>

    </div>
  );
}