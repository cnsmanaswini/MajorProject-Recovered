import React, { useState, useEffect, useRef, useCallback } from 'react'
import { ChevronLeft, ChevronRight, Maximize2, X, Play, Pause } from 'lucide-react'

/**
 * MediaCarousel — Premium multi-media carousel for Mindgram posts.
 *
 * Props:
 *  media       – Array of { url, media_type } objects  (required)
 *  autoPlay    – Boolean: enable auto-advance            (default false)
 *  interval    – Auto-play interval in ms               (default 4000)
 *  aspectRatio – CSS aspect-ratio string                (default '1/1')
 */
export default function MediaCarousel({
  media = [],
  autoPlay = false,
  interval = 4000,
  aspectRatio = '1/1',
}) {
  const [current, setCurrent]         = useState(0)
  const [direction, setDirection]     = useState(1)   // 1 = forward, -1 = backward
  const [animating, setAnimating]     = useState(false)
  const [isHovered, setIsHovered]     = useState(false)
  const [lightbox, setLightbox]       = useState(false)
  const [playing, setPlaying]         = useState(false)

  // Touch / swipe tracking
  const touchStartX = useRef(null)
  const touchStartY = useRef(null)
  const timerRef    = useRef(null)
  const trackRef    = useRef(null)

  const total = media.length

  /* ── helpers ─────────────────────────────── */
  const isVideo = (item) =>
    item?.media_type === 'video' ||
    (item?.url && /\.(mp4|webm|ogg|mov)(\?|$)/i.test(item.url))

  const navigate = useCallback(
    (nextIndex, dir) => {
      if (animating || nextIndex === current) return
      setDirection(dir)
      setAnimating(true)
      setCurrent(nextIndex)
      setTimeout(() => setAnimating(false), 320)
    },
    [animating, current],
  )

  const goPrev = useCallback(() => {
    navigate(current === 0 ? total - 1 : current - 1, -1)
  }, [current, total, navigate])

  const goNext = useCallback(() => {
    navigate(current === total - 1 ? 0 : current + 1, 1)
  }, [current, total, navigate])

  /* ── auto-play ───────────────────────────── */
  useEffect(() => {
    if (!autoPlay || isHovered || total <= 1) return
    timerRef.current = setInterval(goNext, interval)
    return () => clearInterval(timerRef.current)
  }, [autoPlay, isHovered, total, goNext, interval])

  /* ── keyboard ────────────────────────────── */
  useEffect(() => {
    if (!lightbox) return
    const onKey = (e) => {
      if (e.key === 'ArrowLeft')  goPrev()
      if (e.key === 'ArrowRight') goNext()
      if (e.key === 'Escape')     setLightbox(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightbox, goPrev, goNext])

  /* ── touch / swipe ───────────────────────── */
  const onTouchStart = (e) => {
    touchStartX.current = e.touches[0].clientX
    touchStartY.current = e.touches[0].clientY
  }
  const onTouchEnd = (e) => {
    if (touchStartX.current === null) return
    const dx = e.changedTouches[0].clientX - touchStartX.current
    const dy = Math.abs(e.changedTouches[0].clientY - touchStartY.current)
    if (Math.abs(dx) > 40 && dy < 60) {
      dx < 0 ? goNext() : goPrev()
    }
    touchStartX.current = null
  }

  /* ── empty state ─────────────────────────── */
  if (!total) {
    return (
      <div
        className="w-full flex items-center justify-center bg-white/5 rounded-xl text-gray-500 text-sm"
        style={{ aspectRatio }}
      >
        No media
      </div>
    )
  }

  /* ── single item shortcut ────────────────── */
  const renderMedia = (item, idx, inLightbox = false) => {
    if (isVideo(item)) {
      return (
        <video
          key={idx}
          src={item.url}
          className="w-full h-full object-cover"
          controls
          playsInline
          preload="metadata"
          style={inLightbox ? { objectFit: 'contain', maxHeight: '90vh' } : {}}
        />
      )
    }
    return (
      <img
        key={idx}
        src={item.url}
        alt={`Slide ${idx + 1}`}
        className="w-full h-full object-cover"
        loading="lazy"
        draggable={false}
        style={inLightbox ? { objectFit: 'contain', maxHeight: '90vh' } : {}}
      />
    )
  }

  /* ── slide transform ─────────────────────── */
  const slideStyle = (idx) => {
    if (idx === current) return { transform: 'translateX(0%)', zIndex: 2, opacity: 1 }
    // outgoing slide
    const offset = direction * -100
    return { transform: `translateX(${offset}%)`, zIndex: 1, opacity: 0 }
  }

  return (
    <>
      {/* ── Carousel ─────────────────────────── */}
      <div
        ref={trackRef}
        className="relative w-full overflow-hidden bg-gray-950 select-none"
        style={{ aspectRatio }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onTouchStart={onTouchStart}
        onTouchEnd={onTouchEnd}
      >
        {/* Slides */}
        {media.map((item, idx) => (
          <div
            key={idx}
            className="absolute inset-0 transition-all"
            style={{
              ...slideStyle(idx),
              transition: animating ? 'transform 0.32s cubic-bezier(0.4,0,0.2,1), opacity 0.32s ease' : 'none',
            }}
          >
            {renderMedia(item, idx)}
          </div>
        ))}

        {/* Gradient overlays */}
        <div className="absolute inset-x-0 bottom-0 h-20 pointer-events-none"
          style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.55) 0%, transparent 100%)' }}
        />
        {total > 1 && (
          <div className="absolute inset-x-0 top-0 h-14 pointer-events-none"
            style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.4) 0%, transparent 100%)' }}
          />
        )}

        {/* Nav buttons — show on hover */}
        {total > 1 && (
          <>
            <button
              type="button"
              onClick={goPrev}
              aria-label="Previous"
              className="carousel-nav-btn left-2"
              style={{
                opacity: isHovered ? 1 : 0,
                transform: `translateY(-50%) scale(${isHovered ? 1 : 0.85})`,
              }}
            >
              <ChevronLeft size={20} strokeWidth={2.5} />
            </button>
            <button
              type="button"
              onClick={goNext}
              aria-label="Next"
              className="carousel-nav-btn right-2"
              style={{
                opacity: isHovered ? 1 : 0,
                transform: `translateY(-50%) scale(${isHovered ? 1 : 0.85})`,
              }}
            >
              <ChevronRight size={20} strokeWidth={2.5} />
            </button>
          </>
        )}

        {/* Expand button */}
        {!isVideo(media[current]) && (
          <button
            type="button"
            onClick={() => setLightbox(true)}
            aria-label="Expand"
            className="absolute top-2.5 left-2.5 z-20 w-8 h-8 rounded-full
                       bg-black/50 backdrop-blur-sm text-white flex items-center justify-center
                       transition-all duration-200 hover:bg-black/70 hover:scale-110"
            style={{ opacity: isHovered ? 1 : 0 }}
          >
            <Maximize2 size={14} />
          </button>
        )}

        {/* Counter badge */}
        {total > 1 && (
          <div className="absolute top-2.5 right-2.5 z-20
                          px-2.5 py-0.5 rounded-full
                          bg-black/55 backdrop-blur-sm
                          text-white text-xs font-mono font-medium tracking-wide">
            {current + 1}&thinsp;/&thinsp;{total}
          </div>
        )}

        {/* Dot indicators */}
        {total > 1 && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20
                          flex items-center gap-1.5">
            {media.map((_, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => navigate(idx, idx > current ? 1 : -1)}
                aria-label={`Go to slide ${idx + 1}`}
                className="rounded-full bg-white transition-all duration-300"
                style={{
                  width:   idx === current ? '20px' : '6px',
                  height:  '6px',
                  opacity: idx === current ? 1 : 0.45,
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── Lightbox ─────────────────────────── */}
      {lightbox && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.92)', backdropFilter: 'blur(12px)' }}
          onClick={() => setLightbox(false)}
        >
          {/* Close */}
          <button
            type="button"
            onClick={() => setLightbox(false)}
            aria-label="Close lightbox"
            className="absolute top-4 right-4 z-10 w-10 h-10 rounded-full bg-white/10
                       hover:bg-white/20 text-white flex items-center justify-center
                       transition-all duration-200"
          >
            <X size={20} />
          </button>

          {/* Counter */}
          {total > 1 && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2
                            px-3 py-1 rounded-full bg-white/10 text-white text-sm font-mono">
              {current + 1} / {total}
            </div>
          )}

          {/* Image */}
          <div
            className="relative max-w-5xl max-h-[90vh] w-full mx-6 rounded-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {renderMedia(media[current], current, true)}
          </div>

          {/* Nav in lightbox */}
          {total > 1 && (
            <>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); goPrev() }}
                aria-label="Previous"
                className="absolute left-4 top-1/2 -translate-y-1/2
                           w-12 h-12 rounded-full bg-white/10 hover:bg-white/20
                           text-white flex items-center justify-center transition-all"
              >
                <ChevronLeft size={24} />
              </button>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); goNext() }}
                aria-label="Next"
                className="absolute right-4 top-1/2 -translate-y-1/2
                           w-12 h-12 rounded-full bg-white/10 hover:bg-white/20
                           text-white flex items-center justify-center transition-all"
              >
                <ChevronRight size={24} />
              </button>
            </>
          )}

          {/* Dots */}
          {total > 1 && (
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-2">
              {media.map((_, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={(e) => { e.stopPropagation(); navigate(idx, idx > current ? 1 : -1) }}
                  className="rounded-full bg-white transition-all duration-300"
                  style={{ width: idx === current ? '24px' : '8px', height: '8px', opacity: idx === current ? 1 : 0.4 }}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </>
  )
}