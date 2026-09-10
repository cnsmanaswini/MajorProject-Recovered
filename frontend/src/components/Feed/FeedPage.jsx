import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Heart, MessageCircle, Share2, Bookmark, Send, Image, Video, RefreshCw, MapPin, Search, X, Layers, MoreHorizontal, Flag, EyeOff, Trash2 } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import clsx from 'clsx'
import { useAuth } from '../../context/AuthContext'
import { EmotionBadge, RiskBadge, SentimentBar, InterventionBanner } from '../Common/Badges.jsx'
import MediaCarousel from '../Common/MediaCarousel.jsx'
import StoryUploader from '../StoryUploader.jsx'
import StoryRing from '../StoryRing.jsx'
import StoryViewer from '../StoryViewersModal.jsx'

// ── API calls ─────────────────────────────────────────────────
const useFeedApi = () => {
  const { api } = useAuth()
  return {
    getFeed:        (explore) => api.get(explore ? '/feed/explore' : '/feed'),
    createPost:     (data)    => api.post('/posts', data),
    likePost:       (id)      => api.post(`/posts/${id}/like`),
    getComments:    (id)      => api.get(`/interactions/comments/${id}`),
    addComment:     (data)    => api.post('/interactions/comment', data),
    getAgentStatus: ()        => api.get(`/agents/status/me`),
  }
}

// ── Post Card ─────────────────────────────────────────────────
const getPostMedia = (post) => {
  if (post.media?.length) {
    return [...post.media].sort((a, b) => (a.position || 0) - (b.position || 0))
  }
  if (post.image_url) return [{ id: 'legacy-image', media_type: 'image', url: post.image_url, position: 0 }]
  if (post.video_url) return [{ id: 'legacy-video', media_type: 'video', url: post.video_url, position: 0 }]
  return []
}


function usePostImpression(postId, userId, enabled = true) {
  const cardRef = useRef(null)
  const visibleSinceRef = useRef(null)
  const sentRef = useRef(false)
  const { api } = useAuth()

  const sendImpression = useCallback((dwellMs) => {
    if (!enabled || !userId || sentRef.current || dwellMs < 0) return
    sentRef.current = true
    api.post('/interactions/impression', {
      user_id: userId,
      post_id: postId,
      dwell_ms: dwellMs,
    }).catch(() => {})
  }, [api, enabled, postId, userId])

  useEffect(() => {
    if (!enabled || !userId) return undefined
    const node = cardRef.current
    if (!node) return undefined

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          visibleSinceRef.current = Date.now()
        } else if (visibleSinceRef.current) {
          const dwellMs = Date.now() - visibleSinceRef.current
          visibleSinceRef.current = null
          sendImpression(dwellMs)
        }
      },
      { threshold: 0.5 },
    )

    observer.observe(node)
    return () => {
      if (visibleSinceRef.current) {
        sendImpression(Date.now() - visibleSinceRef.current)
      }
      observer.disconnect()
    }
  }, [enabled, postId, sendImpression, userId])

  return cardRef
}


function PostCard({ post, onLike, onHide, trackImpressions = true }) {
  const { user, api } = useAuth()
  const navigate = useNavigate()
  const cardRef = usePostImpression(post.id, user?.id, trackImpressions)

  const goToProfile = () => {
    if (post.author?.username) navigate(`/profile/${post.author.username}`)
  }
  const goToCommenter = (c) => {
    navigate(`/profile/${c.user?.username || c.user_id}`)
  }
  const [liked, setLiked]           = useState(false)
  const [likeCount, setLikeCount]   = useState(post.likes_count || 0)
  const [showComments, setShowComments] = useState(false)
  const [comments, setComments]     = useState([])
  const [newComment, setNewComment] = useState('')
  const [showAI, setShowAI]         = useState(false)
  const [posting, setPosting]       = useState(false)
  const [showMenu, setShowMenu]       = useState(false)
  const [showReport, setShowReport]   = useState(false)
  const [reportReason, setReportReason] = useState('other')
  const [reportDetails, setReportDetails] = useState('')
  const [feedbackMsg, setFeedbackMsg] = useState('')
  const media = getPostMedia(post)

  const handleLike = async () => {
    const prevLiked = liked
    const prevCount = likeCount
    setLiked(!prevLiked)
    setLikeCount(c => !prevLiked ? c + 1 : c - 1)
    try {
      const res = await api.post(`/posts/${post.id}/like`)
      setLiked(res.data.is_liked)
      setLikeCount(res.data.likes_count)
    } catch {
      setLiked(prevLiked)
      setLikeCount(prevCount)
    }
  }

  const loadComments = async () => {
    try {
      const res = await api.get(`/interactions/comments/${post.id}`)
      setComments(res.data)
    } catch {
      setComments([])
    }
  }

  const handleCommentToggle = () => {
    setShowComments(s => !s)
    if (!showComments) loadComments()
  }

  const submitComment = async (e) => {
    e.preventDefault()
    if (!newComment.trim() || posting) return
    setPosting(true)
    try {
      await api.post('/interactions/comment', {
        post_id: post.id,
        user_id: user.id,
        content: newComment,
      })
      setNewComment('')
      loadComments()
    } catch {}
    setPosting(false)
  }

  const handleNotInterested = async () => {
    setShowMenu(false)
    try {
      await api.post('/interactions', {
        user_id: user.id,
        post_id: post.id,
        action: 'not_interested',
      })
      setFeedbackMsg('We will show less content like this.')
      onHide?.(post.id)
    } catch {
      setFeedbackMsg('Could not save your preference.')
    }
  }

  const handleDelete = async () => {
    setShowMenu(false)
    if (!window.confirm('Delete this post? This cannot be undone.')) return
    try {
      await api.delete(`/posts/${post.id}`)
      onHide?.(post.id)
    } catch {
      setFeedbackMsg('Could not delete post.')
    }
  }

  const handleReport = async (e) => {
    e.preventDefault()
    try {
      await api.post('/interactions/report', {
        user_id: user.id,
        post_id: post.id,
        reason: reportReason,
        details: reportDetails.trim() || undefined,
      })
      setShowReport(false)
      setShowMenu(false)
      setReportDetails('')
      setFeedbackMsg('Thanks for reporting. We will review this post.')
      onHide?.(post.id)
    } catch {
      setFeedbackMsg('Could not submit report.')
    }
  }

  return (
    <article ref={cardRef} className="card animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between p-4 pb-3">
        <div className="flex items-center gap-3">
          <div className="story-ring cursor-pointer" onClick={goToProfile}>
            <img
              src={post.author?.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${post.author?.username}`}
              alt=""
              className="w-9 h-9 rounded-full bg-gray-800"
            />
          </div>
          <div>
            <p className="font-semibold text-sm text-white">{post.author?.display_name}</p>
            <div className="flex items-center gap-1.5 text-xs text-gray-500">
              <span className="cursor-pointer hover:text-gray-300 transition-colors" onClick={goToProfile}>@{post.author?.username}</span>
              {post.location && (
                <>
                  <span>·</span>
                  <MapPin size={10} />
                  <span>{post.location}</span>
                </>
              )}
              <span>·</span>
              <span>{formatDistanceToNow(new Date(post.created_at), { addSuffix: true })}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {post.sarcasm && (
            <span className="pill bg-purple-500/15 text-purple-300 border border-purple-500/20 text-[10px]">
              🎭 sarcasm
            </span>
          )}
          <EmotionBadge emotion={post.emotion} score={post.emotion_score} />
          <div className="relative">
            <button
              type="button"
              onClick={() => setShowMenu(s => !s)}
              className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/10 transition-all"
              aria-label="Post options"
            >
              <MoreHorizontal size={16} />
            </button>
            {showMenu && (
              <div className="absolute right-0 mt-1 z-20 w-48 card p-1 shadow-xl">
                <button
                  type="button"
                  onClick={handleNotInterested}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-white/5"
                >
                  <EyeOff size={14} />
                  Not interested
                </button>
                <button
                  type="button"
                  onClick={() => { setShowReport(true); setShowMenu(false) }}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-red-300 hover:bg-red-500/10"
                >
                  <Flag size={14} />
                  Report post
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {feedbackMsg && (
        <p className="px-4 pb-2 text-xs text-brand-300">{feedbackMsg}</p>
      )}

      {showReport && (
        <form onSubmit={handleReport} className="mx-4 mb-3 p-3 rounded-xl bg-black/30 border border-white/10 space-y-2">
          <p className="text-xs text-gray-400">Why are you reporting this post?</p>
          <select
            value={reportReason}
            onChange={e => setReportReason(e.target.value)}
            className="input-field text-sm w-full"
          >
            <option value="spam">Spam</option>
            <option value="harassment">Harassment</option>
            <option value="self_harm">Self-harm concern</option>
            <option value="other">Other</option>
          </select>
          <textarea
            value={reportDetails}
            onChange={e => setReportDetails(e.target.value)}
            placeholder="Optional details..."
            rows={2}
            className="input-field text-sm w-full resize-none"
          />
          <div className="flex gap-2 justify-end">
            <button
              type="button"
              onClick={() => setShowReport(false)}
              className="btn-ghost py-1.5 px-3 text-xs"
            >
              Cancel
            </button>
            <button type="submit" className="btn-primary py-1.5 px-3 text-xs">
              Submit report
            </button>
          </div>
        </form>
      )}

      {/* Image — square like Instagram */}
      <MediaCarousel media={media} />

      {/* Actions */}
      <div className="flex items-center gap-1 px-3 py-2">
        <button
          onClick={handleLike}
          className={clsx(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm transition-all',
            liked
              ? 'text-red-400 bg-red-400/10'
              : 'text-gray-400 hover:text-red-400 hover:bg-red-400/10'
          )}
        >
          <Heart size={16} fill={liked ? 'currentColor' : 'none'} />
          <span>{likeCount}</span>
        </button>

        <button
          onClick={handleCommentToggle}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm
                     text-gray-400 hover:text-brand-300 hover:bg-brand-500/10 transition-all"
        >
          <MessageCircle size={16} />
          <span>{post.comments_count || 0}</span>
        </button>

        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm
                           text-gray-400 hover:text-green-400 hover:bg-green-400/10 transition-all">
          <Share2 size={16} />
        </button>

        <button className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm
                           text-gray-400 hover:text-yellow-400 hover:bg-yellow-400/10 transition-all">
          <Bookmark size={16} />
        </button>

        <button
          onClick={() => setShowAI(s => !s)}
          className={clsx(
            'flex items-center gap-1 px-2 py-1.5 rounded-xl text-xs font-mono transition-all',
            showAI ? 'bg-brand-500/20 text-brand-300' : 'text-gray-600 hover:text-brand-400'
          )}
        >
          AI
        </button>
      </div>

      {/* Caption — Instagram style: username + text */}
      {post.content && (
        <div className="px-4 pb-2">
          <p className="text-sm text-gray-200 leading-relaxed">
            <span
              className="font-semibold text-white mr-1.5 cursor-pointer hover:underline"
              onClick={goToProfile}
            >
              {post.author?.username}
            </span>
            {post.content}
          </p>
          {post.topics?.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {post.topics.filter(t => !t.startsWith('emotion:') && !t.startsWith('loc:')).map(topic => (
                <span key={topic} className="text-[10px] text-brand-400/80">#{topic}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* AI Panel */}
      {showAI && (
        <div className="mx-4 mb-3 p-3 rounded-xl bg-black/30 border border-white/10
                        space-y-2 animate-fade-in">
          <p className="text-[10px] text-gray-500 uppercase tracking-widest font-mono">
            AI Analysis
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <p className="text-gray-500 mb-1">Sentiment</p>
              <SentimentBar score={post.sentiment_score || 0} />
            </div>
            <div>
              <p className="text-gray-500 mb-1">Risk Score</p>
              <div className="flex items-center gap-1.5">
                <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={clsx(
                      'h-full rounded-full',
                      (post.risk_score || 0) > 0.6 ? 'bg-red-400'
                      : (post.risk_score || 0) > 0.35 ? 'bg-yellow-400'
                      : 'bg-green-400'
                    )}
                    style={{ width: `${(post.risk_score || 0) * 100}%` }}
                  />
                </div>
                <span className="text-gray-400 w-8 text-right">
                  {Math.round((post.risk_score || 0) * 100)}%
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
            <span>Feed score:</span>
            <span className="text-brand-300 font-mono">
              {(post.feed_score || 0).toFixed(3)}
            </span>
          </div>
        </div>
      )}

      {/* Comments */}
      {showComments && (
        <div className="border-t border-white/10 animate-fade-in">
          <div className="max-h-48 overflow-y-auto p-4 space-y-3">
            {comments.length === 0 && (
              <p className="text-gray-500 text-xs text-center py-2">
                No comments yet. Be the first!
              </p>
            )}
            {comments.map(c => (
              <div key={c.id} className="flex gap-2">
                <img
                  src={c.user?.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${c.user_id}`}
                  alt=""
                  onClick={() => goToCommenter(c)}
                  className="w-6 h-6 rounded-full bg-gray-800 flex-shrink-0 mt-0.5 cursor-pointer"
                />
                <div className="flex-1 bg-white/5 rounded-xl px-3 py-1.5">
                  <p
                    onClick={() => goToCommenter(c)}
                    className="text-xs text-brand-300 font-medium cursor-pointer hover:underline w-fit"
                  >
                    {c.user?.username || `user_${c.user_id}`}
                  </p>
                  <p className="text-xs text-gray-300">{c.content}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="px-4 pb-4">
            <form onSubmit={submitComment} className="flex gap-2">
              <input
                value={newComment}
                onChange={e => setNewComment(e.target.value)}
                placeholder="Add a comment..."
                className="input-field text-sm py-2 flex-1"
              />
              <button
                type="submit"
                disabled={!newComment.trim() || posting}
                className="btn-primary py-2 px-3 disabled:opacity-40"
              >
                <Send size={14} />
              </button>
            </form>
          </div>
        </div>
      )}
    </article>
  )
}

// ── Compose Post ──────────────────────────────────────────────
const MAX_IMAGE_MB = 10
const MAX_VIDEO_MB = 100
const MAX_IMAGES = 10

function ComposePost({ onPost }) {
  const { user, api } = useAuth()
  const [content, setContent]       = useState('')
  const [location, setLocation]     = useState('')
  const [mediaItems, setMediaItems] = useState([])
  const [showExtras, setShowExtras] = useState(false)
  const [posting, setPosting]       = useState(false)
  const [analyzing, setAnalyzing]   = useState(false)
  const [progress, setProgress]     = useState(0)
  const [error, setError]           = useState('')
  const fileRef = useRef()
  const hasMedia = mediaItems.length > 0
  const mediaType = mediaItems[0]?.type || null

  const handleFile = (e) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    setError('')

    const videos = files.filter(file => file.type.startsWith('video/'))
    const images = files.filter(file => file.type.startsWith('image/'))
    if (videos.length && (images.length || videos.length > 1)) {
      setError('Choose up to 10 photos for a carousel or one video for a reel.')
      return
    }
    if (!videos.length && !images.length) {
      setError('Please choose image or video files.')
      return
    }
    if (images.length > MAX_IMAGES) {
      setError(`You can add up to ${MAX_IMAGES} photos per post.`)
      return
    }

    const invalid = files.find(file => {
      const isVideo = file.type.startsWith('video/')
      const maxMb = isVideo ? MAX_VIDEO_MB : MAX_IMAGE_MB
      return file.size / (1024 * 1024) > maxMb
    })
    if (invalid) {
      const isVideo = invalid.type.startsWith('video/')
      const maxMb = isVideo ? MAX_VIDEO_MB : MAX_IMAGE_MB
      setError(`File too large (${(invalid.size / (1024 * 1024)).toFixed(1)}MB). Max is ${maxMb}MB for ${isVideo ? 'videos' : 'images'}.`)
      return
    }

    mediaItems.forEach(item => URL.revokeObjectURL(item.preview))
    setMediaItems(files.map((file, index) => ({
      file,
      preview: URL.createObjectURL(file),
      type: file.type.startsWith('video/') ? 'video' : 'image',
      id: `${file.name}-${file.lastModified}-${index}`,
    })))
  }

  const clearMedia = (id = null) => {
    setMediaItems(items => {
      const removed = id ? items.filter(item => item.id === id) : items
      removed.forEach(item => URL.revokeObjectURL(item.preview))
      return id ? items.filter(item => item.id !== id) : []
    })
    if (fileRef.current) fileRef.current.value = ''
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!content.trim() && !hasMedia) return
    setPosting(true)
    setAnalyzing(true)
    setProgress(0)
    setError('')

    try {
      const formData = new FormData()
      formData.append('content', content)
      formData.append('location', location)

      // IMPORTANT: field name must match what the backend expects —
      // 'image' for images, 'video' for videos. Sending a video under
      // the 'image' key (or vice versa) means the backend silently
      // skips the upload.
      if (mediaType === 'image') {
        mediaItems.forEach(item => formData.append('images', item.file))
      } else if (mediaType === 'video') {
        formData.append('video', mediaItems[0].file)
        formData.append('is_reel', 'true')
      }

      const res = await api.post('/posts', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (evt) => {
          if (evt.total) setProgress(Math.round((evt.loaded / evt.total) * 100))
        },
      })
      onPost(res.data)
      setContent('')
      setLocation('')
      clearMedia()
      setShowExtras(false)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || 'Post failed. Please try again.')
    } finally {
      setPosting(false)
      setAnalyzing(false)
      setProgress(0)
    }
  }

  return (
    <div className="card p-4 mb-4">
      <div className="flex gap-3">
        <img
          src={user?.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${user?.username}`}
          alt=""
          className="w-9 h-9 rounded-full bg-gray-800 flex-shrink-0 border-2 border-brand-500/30"
        />
        <div className="flex-1">
          <textarea
            value={content}
            onChange={e => setContent(e.target.value)}
            placeholder={mediaType === 'image' ? 'Write a caption...' : "What's on your mind?"}
            rows={hasMedia ? 2 : 2}
            className="input-field resize-none text-sm w-full"
          />

          {/* Media preview — square crop like Instagram */}
          {hasMedia && mediaType === 'image' && (
            <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2">
              {mediaItems.map((item, index) => (
                <div key={item.id} className="relative rounded-xl overflow-hidden bg-gray-900" style={{ aspectRatio: '1/1' }}>
                  <img src={item.preview} alt="" className="w-full h-full object-cover" />
                  {mediaItems.length > 1 && (
                    <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded-full bg-black/60 text-white text-[10px] font-mono">
                      {index + 1}
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => clearMedia(item.id)}
                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-black/60 text-white
                               flex items-center justify-center text-xs hover:bg-black/80"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {hasMedia && mediaType === 'video' && (
            <div className="relative mt-2 rounded-xl overflow-hidden bg-black">
              <video src={mediaItems[0].preview} className="w-full max-h-64 object-contain" controls playsInline />
              <button
                type="button"
                onClick={() => clearMedia()}
                className="absolute top-2 right-2 w-6 h-6 rounded-full bg-black/60 text-white
                           flex items-center justify-center text-xs hover:bg-black/80"
              >
                <X size={12} />
              </button>
            </div>
          )}

          {/* Upload progress */}
          {posting && progress > 0 && progress < 100 && (
            <div className="mt-2 h-1 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-brand-500 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          {/* Error */}
          {error && (
            <p className="text-xs text-red-400 mt-2">{error}</p>
          )}

          {/* Location input */}
          {showExtras && (
            <input
              value={location}
              onChange={e => setLocation(e.target.value)}
              placeholder="Add location..."
              className="input-field text-sm mt-2 w-full"
            />
          )}

          <div className="flex items-center justify-between mt-2">
            <div className="flex gap-1">
              <input
                ref={fileRef}
                type="file"
                accept="image/*,video/*"
                multiple
                onChange={handleFile}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileRef.current.click()}
                className="btn-ghost py-1.5 px-2 text-xs flex items-center gap-1"
              >
                {mediaType === 'video' ? <Video size={14} /> : mediaItems.length > 1 ? <Layers size={14} /> : <Image size={14} />} Media
              </button>
              <button
                type="button"
                onClick={() => setShowExtras(s => !s)}
                className="btn-ghost py-1.5 px-2 text-xs flex items-center gap-1"
              >
                <MapPin size={14} /> Location
              </button>
            </div>

            <button
              onClick={handleSubmit}
              disabled={(!content.trim() && !hasMedia) || posting}
              className="btn-primary py-1.5 px-4 text-sm disabled:opacity-40
                         flex items-center gap-1.5"
            >
              {analyzing ? (
                <>
                  <RefreshCw size={12} className="animate-spin" />
                  {progress > 0 && progress < 100 ? `Uploading ${progress}%` : 'Analyzing...'}
                </>
              ) : (
                <>
                  <Send size={12} />
                  Post
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Trending Topics Bar ───────────────────────────────────────
function TrendingBar({ topics }) {
  if (!topics?.length) return null
  return (
    <div className="mb-4">
      <p className="text-[10px] uppercase tracking-widest text-gray-500 font-mono mb-2">
        Trending now
      </p>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
        {topics.map(({ topic, score }) => (
          <span
            key={topic}
            className="pill bg-brand-500/10 text-brand-300 border border-brand-500/20 text-xs whitespace-nowrap"
          >
            #{topic.replace(/^emotion:/, '').replace(/^loc:/, '')}
            <span className="ml-1 text-gray-500 font-mono text-[10px]">
              {Math.round(score * 100)}%
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}


// ── Feed Page ─────────────────────────────────────────────────
export default function FeedPage({ explore = false }) {
  const { api, user }   = useAuth()
  const navigate         = useNavigate()
  const [posts, setPosts]           = useState([])
  const [agentStatus, setAgentStatus] = useState(null)
  const [loading, setLoading]       = useState(true)
  const [page, setPage]             = useState(0)
  const [hasMore, setHasMore]       = useState(true)
  const [search, setSearch]               = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [trendingTopics, setTrendingTopics] = useState([])
  const [followingIds, setFollowingIds] = useState([])
  const [showStoryUploader, setShowStoryUploader] = useState(false)
  const [viewingUserId, setViewingUserId] = useState(null)
  const [storyRingKey, setStoryRingKey] = useState(0)

  // Search users (Explore page only)
  useEffect(() => {
    if (!explore || !search.trim()) { setSearchResults([]); return }
    const timer = setTimeout(() => {
      api.get(`/users/search?q=${search}`)
        .then(r => setSearchResults(r.data))
        .catch(() => {})
    }, 400)
    return () => clearTimeout(timer)
  }, [search, explore])

  useEffect(() => {
    if (!user?.id || explore) {
      setFollowingIds([])
      return
    }
    api.get(`/users/${user.id}/following`)
      .then(r => setFollowingIds((r.data || []).map(u => u.id)))
      .catch(() => setFollowingIds([]))
  }, [user?.id, explore, api])

  const load = useCallback(async (reset = false) => {
    setLoading(true)
    try {
      const offset = reset ? 0 : page * 20
      const [feedRes, agentRes, trendingRes] = await Promise.all([
        api.get(explore ? '/feed/explore' : `/feed?limit=20&offset=${offset}`),
        api.get(`/agents/status/${user?.id}`).catch(() => ({ data: null })),
        api.get('/feed/trending').catch(() => ({ data: [] })),
      ])
      const newPosts = feedRes.data || []
      setPosts(prev => reset ? newPosts : [...prev, ...newPosts])
      setAgentStatus(agentRes.data)
      setTrendingTopics(trendingRes.data || [])
      setHasMore(newPosts.length === 20)
      if (!reset) setPage(p => p + 1)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [page, explore, user])

  useEffect(() => {
    load(true)
  }, [explore])

  const handleNewPost = (post) => {
    setPosts(prev => [post, ...prev])
  }

  const handleHidePost = (postId) => {
    setPosts(prev => prev.filter(p => p.id !== postId))
  }

  return (
    <div className="max-w-xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-display text-2xl text-white">
          {explore ? 'Explore' : 'Feed'}
        </h1>
        <button
          onClick={() => load(true)}
          className="btn-ghost py-1.5 px-3 text-sm flex items-center gap-1.5"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Search people (Explore only) */}
      {explore && (
        <div className="relative mb-6">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search people..."
            className="input-field pl-8 text-sm py-2.5 w-full"
          />
          {searchResults.length > 0 && (
            <div className="absolute z-30 mt-1 w-full card p-1 max-h-64 overflow-y-auto">
              {searchResults.map(u => (
                <button
                  key={u.id}
                  onClick={() => {
                    setSearch('')
                    setSearchResults([])
                    navigate(`/profile/${u.username}`)
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-all text-left"
                >
                  <img
                    src={u.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${u.username}`}
                    alt=""
                    className="w-8 h-8 rounded-full bg-gray-800"
                  />
                  <div className="min-w-0">
                    <p className="text-sm text-white truncate">{u.display_name}</p>
                    <p className="text-xs text-gray-500 truncate">@{u.username}</p>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}


      {/* Stories */}
      {!explore && user && (
        <div className="mb-4">
          <StoryRing
            currentUserId={user.id}
            currentUser={user}
            followingIds={followingIds}
            refreshKey={storyRingKey}
            onOpenUploader={() => setShowStoryUploader(true)}
            onOpenViewer={(userId) => setViewingUserId(userId)}
          />
        </div>
      )}

      {showStoryUploader && (
        <StoryUploader
          onClose={() => setShowStoryUploader(false)}
          onUploaded={() => {
            setShowStoryUploader(false)
            setStoryRingKey(k => k + 1)
          }}
        />
      )}

      {viewingUserId && (
        <StoryViewer
          userId={viewingUserId}
          onClose={() => {
            setViewingUserId(null)
            setStoryRingKey(k => k + 1)
          }}
        />
      )}

      {!explore && <TrendingBar topics={trendingTopics} />}

      {/* Compose */}
      {!explore && <ComposePost onPost={handleNewPost} />}

      {/* Intervention banner — shown silently when AI detects risk */}
      <InterventionBanner agentStatus={agentStatus} />

      {/* Posts */}
      {loading && posts.length === 0 ? (
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="card p-4 animate-pulse space-y-3">
              <div className="flex gap-3">
                <div className="w-10 h-10 rounded-full bg-white/10" />
                <div className="flex-1 space-y-2">
                  <div className="h-3 bg-white/10 rounded w-1/3" />
                  <div className="h-2 bg-white/10 rounded w-1/4" />
                </div>
              </div>
              <div className="h-40 bg-white/5 rounded-xl" />
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="space-y-4">
            {posts.map(post => (
              <PostCard
                key={post.id}
                post={post}
                onHide={handleHidePost}
                trackImpressions={!explore}
              />
            ))}
          </div>

          {/* Load more */}
          {hasMore && !loading && (
            <button
              onClick={() => load(false)}
              className="w-full mt-4 btn-ghost py-3 text-sm"
            >
              Load more
            </button>
          )}

          {posts.length === 0 && !loading && (
            <div className="text-center py-12">
              <p className="text-gray-400 text-sm">
                {explore
                  ? 'No posts yet. Be the first!'
                  : 'Follow people to see their posts here!'}
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}