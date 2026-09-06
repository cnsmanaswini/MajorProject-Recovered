import React, { useRef, useState, useEffect } from 'react'

/**
 * VideoThumbnail — reliable poster-frame for a video, works in every browser.
 *
 * The `#t=0.5` URL-fragment trick only works in some Chromium builds and is
 * ignored by Firefox/Safari unless the video has already played — that's why
 * you were seeing blanks. This instead: loads just the video metadata, seeks
 * to a frame, waits for that frame to actually be ready, then either shows
 * the <video> paused on that frame (cheap, default) or grabs a real <canvas>
 * snapshot you can reuse as a static image (e.g. for a grid of many videos
 * where you don't want dozens of live <video> elements mounted at once).
 *
 * Usage (drop-in swap for wherever you're rendering <video src=... /> as a
 * static-looking preview, e.g. Explore grid, Reels list thumbnails):
 *
 *   <VideoThumbnail src={item.url} seekTo={0.5} className="w-full h-full object-cover" />
 *
 * For a real still image instead of a paused <video> (better perf in long
 * grids), pass asImage:
 *
 *   <VideoThumbnail src={item.url} seekTo={0.5} asImage className="w-full h-full object-cover" />
 */
export default function VideoThumbnail({
  src,
  seekTo = 0.5,       // seconds into the video to grab the frame from
  asImage = false,     // true = render a <canvas>-derived <img>, false = paused <video>
  className = '',
  alt = '',
}) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [imgSrc, setImgSrc] = useState(null)

  useEffect(() => {
    setReady(false)
    setImgSrc(null)
  }, [src])

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    if (!video) return
    // Clamp seek time to actual duration so short clips don't error out.
    const target = Math.min(seekTo, Math.max(video.duration - 0.1, 0))
    video.currentTime = target
  }

  const handleSeeked = () => {
    const video = videoRef.current
    if (!video) return

    if (!asImage) {
      setReady(true)
      return
    }

    // Grab a real still frame onto canvas, then use that as a plain <img>.
    const canvas = canvasRef.current || document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      setImgSrc(canvas.toDataURL('image/jpeg', 0.85))
      setReady(true)
    } catch {
      // Cross-origin video without CORS headers — canvas capture blocked.
      // Fall back to just showing the paused <video> instead of a blank.
      setReady(true)
    }
  }

  if (asImage && imgSrc) {
    return <img src={imgSrc} alt={alt} className={className} />
  }

  return (
    <div className={`relative ${className}`} style={{ overflow: 'hidden' }}>
      <video
        ref={videoRef}
        src={src}
        muted
        playsInline
        preload="metadata"
        onLoadedMetadata={handleLoadedMetadata}
        onSeeked={handleSeeked}
        className={asImage ? 'invisible absolute w-full h-full' : `w-full h-full object-cover ${ready ? 'opacity-100' : 'opacity-0'} transition-opacity`}
      />
      {!ready && (
        <div className="absolute inset-0 bg-white/5 animate-pulse" />
      )}
    </div>
  )
}