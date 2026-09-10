import React, { useState, useEffect, useRef } from 'react'
import { useLocation } from 'react-router-dom'
import { Send, Search, Circle, MessageCircle } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import clsx from 'clsx'
import { useAuth } from '../../context/AuthContext.jsx'
import { useSocket } from '../../context/SocketContext.jsx'
import { EmotionBadge } from '../Common/Badges.jsx'

export default function MessagesPage() {
  const { user, api }               = useAuth()
  const { sendMessage, on, isConnected } = useSocket()
  const location = useLocation()
  const [conversations, setConversations] = useState([])
  const [selected, setSelected]     = useState(location.state?.selectedUser || null)
  const [thread, setThread]         = useState([])
  const [input, setInput]           = useState('')
  const [search, setSearch]         = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [loading, setLoading]       = useState(false)
  const [mutual, setMutual]         = useState(null)
  const [wsError, setWsError]       = useState('')
  const bottomRef = useRef()

  // Load conversations
  useEffect(() => {
    api.get('/messages/conversations')
      .then(r => setConversations(r.data))
      .catch(() => {})
  }, [])

  // Load thread when selecting conversation
  // Backend route is GET /messages/thread/{user_id}/{other_user_id}
  useEffect(() => {
    if (!selected) return
    setLoading(true)
    api.get(`/messages/thread/${user?.id}/${selected.id}`)
      .then(r => setThread(r.data))
      .catch(() => setThread([]))
      .finally(() => setLoading(false))
  }, [selected, user?.id])

  // Check mutual-follow status when selecting a conversation
  useEffect(() => {
    if (!selected) { setMutual(null); return }
    setMutual(null)
    setWsError('')
    api.get(`/users/${selected.username}`)
      .then(r => setMutual(r.data.is_following && r.data.follows_you))
      .catch(() => setMutual(false))
  }, [selected])

  // Listen for server-side rejection (e.g. not mutual followers)
  useEffect(() => {
    const cleanup = on('error', (data) => {
      setWsError(data.detail || 'Message could not be sent.')
    })
    return cleanup
  }, [])

  // Listen for new messages via WebSocket
  useEffect(() => {
    const cleanup = on('new_message', (msg) => {
      if (selected && (msg.sender_id === selected.id || msg.receiver_id === selected.id)) {
        setThread(prev => [...prev, msg])
      }
      // Update conversations list
      setConversations(prev => {
        const partnerId = msg.sender_id === user?.id ? msg.receiver_id : msg.sender_id
        const existing = prev.find(c => c.user.id === partnerId)
        if (existing) {
          return prev.map(c => c.user.id === partnerId
            ? { ...c, last_message: { content: msg.content, created_at: msg.created_at, is_mine: msg.sender_id === user?.id, sentiment: msg.sentiment } }
            : c
          )
        }
        return prev
      })
    })
    const cleanup2 = on('message_sent', (msg) => {
      setThread(prev => {
        const exists = prev.find(m => m.id === msg.id)
        return exists ? prev : [...prev, msg]
      })
    })
    return () => { cleanup(); cleanup2() }
  }, [selected, user])

  // Scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [thread])

  // Search users
  useEffect(() => {
    if (!search.trim()) { setSearchResults([]); return }
    const timer = setTimeout(() => {
      api.get(`/users/search?q=${search}`)
        .then(r => setSearchResults(r.data))
        .catch(() => {})
    }, 400)
    return () => clearTimeout(timer)
  }, [search])

  const handleSend = (e) => {
    e.preventDefault()
    if (!input.trim() || !selected || !mutual) return
    sendMessage(selected.id, input.trim())
    setInput('')
  }

  const SENTIMENT_BORDER = {
    positive: 'border-l-2 border-green-500/40',
    negative: 'border-l-2 border-red-500/40',
    neutral:  '',
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div className="w-72 flex-shrink-0 border-r border-white/10 flex flex-col bg-black/20">
        {/* Header */}
        <div className="p-4 border-b border-white/10">
          <h2 className="font-display text-xl text-white mb-3">Messages</h2>
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search people..."
              className="input-field pl-8 text-sm py-2 w-full"
            />
          </div>
        </div>

        {/* Search results */}
        {searchResults.length > 0 && (
          <div className="border-b border-white/10 max-h-48 overflow-y-auto">
            {searchResults.map(u => (
              <button
                key={u.id}
                onClick={() => {
                  setSelected(u)
                  setSearch('')
                  setSearchResults([])
                }}
                className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-white/5 transition-all"
              >
                <img
                  src={u.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${u.username}`}
                  alt=""
                  className="w-8 h-8 rounded-full bg-gray-800"
                />
                <div className="text-left">
                  <p className="text-sm font-medium text-white">{u.display_name}</p>
                  <p className="text-xs text-gray-500">@{u.username}</p>
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Conversations */}
        <div className="flex-1 overflow-y-auto">
          {conversations.length === 0 && (
            <p className="text-center text-gray-500 text-xs py-8">
              No conversations yet. Search for someone to chat!
            </p>
          )}
          {conversations.map(conv => (
            <button
              key={conv.user.id}
              onClick={() => setSelected(conv.user)}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-3 transition-all text-left border-l-2',
                selected?.id === conv.user.id
                  ? 'bg-brand-500/10 border-brand-500'
                  : 'border-transparent hover:bg-white/5'
              )}
            >
              <div className="relative flex-shrink-0">
                <img
                  src={conv.user.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${conv.user.username}`}
                  alt=""
                  className="w-10 h-10 rounded-full bg-gray-800"
                />
                {conv.is_online && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 rounded-full bg-green-400 border-2 border-black" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-white truncate">{conv.user.display_name}</p>
                  {conv.last_message && (
                    <span className="text-[10px] text-gray-500 flex-shrink-0 ml-1">
                      {formatDistanceToNow(new Date(conv.last_message.created_at), { addSuffix: false })}
                    </span>
                  )}
                </div>
                {conv.last_message && (
                  <p className="text-xs text-gray-500 truncate">
                    {conv.last_message.is_mine ? 'You: ' : ''}{conv.last_message.content}
                  </p>
                )}
              </div>
            </button>
          ))}
        </div>

        {/* Connection status */}
        <div className="p-3 border-t border-white/10 flex items-center gap-2">
          <Circle
            size={8}
            className={isConnected ? 'text-green-400 fill-green-400' : 'text-red-400 fill-red-400'}
          />
          <span className="text-[10px] text-gray-500 font-mono">
            {isConnected ? 'Live chat connected' : 'Reconnecting...'}
          </span>
        </div>
      </div>

      {/* Thread */}
      <div className="flex-1 flex flex-col min-w-0">
        {selected ? (
          <>
            {/* Header */}
            <div className="flex items-center gap-3 p-4 border-b border-white/10 bg-black/20">
              <img
                src={selected.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${selected.username}`}
                alt=""
                className="w-9 h-9 rounded-full bg-gray-800 border-2 border-brand-500/30"
              />
              <div>
                <p className="font-medium text-sm text-white">{selected.display_name}</p>
                <p className="text-xs text-gray-500">@{selected.username}</p>
              </div>
              <span className="ml-auto text-[10px] text-gray-600 font-mono bg-white/5 px-2 py-1 rounded-lg">
                sentiment-aware chat
              </span>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {loading && (
                <div className="flex justify-center py-4">
                  <div className="w-5 h-5 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
                </div>
              )}
              {thread.map((msg, i) => {
                const isMine = msg.sender_id === user?.id
                return (
                  <div key={msg.id || i} className={clsx('flex', isMine ? 'justify-end' : 'justify-start')}>
                    <div className={clsx(
                      'max-w-xs px-4 py-2.5 rounded-2xl text-sm',
                      SENTIMENT_BORDER[msg.sentiment] || '',
                      isMine
                        ? 'bg-brand-500/20 text-gray-100 rounded-br-sm'
                        : 'bg-white/8 text-gray-200 rounded-bl-sm'
                    )}>
                      <p className="leading-relaxed">{msg.content}</p>
                      <div className="flex items-center gap-2 mt-1">
                        {msg.emotion && msg.emotion !== 'neutral' && (
                          <EmotionBadge emotion={msg.emotion} size="sm" />
                        )}
                        <span className="text-[10px] text-gray-500 ml-auto">
                          {formatDistanceToNow(new Date(msg.created_at), { addSuffix: true })}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
              <div ref={bottomRef} />
            </div>

            {/* Input */}
            {mutual === false ? (
              <div className="p-4 border-t border-white/10 bg-black/20 text-center">
                <p className="text-xs text-gray-500">
                  You can only message users who follow you back and who you follow too.
                </p>
              </div>
            ) : (
              <div className="p-4 border-t border-white/10 bg-black/20">
                {wsError && (
                  <p className="text-xs text-red-400 mb-2 text-center">{wsError}</p>
                )}
                <form onSubmit={handleSend} className="flex gap-2">
                  <input
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder="Type a message..."
                    disabled={mutual !== true}
                    className="input-field text-sm flex-1 disabled:opacity-50"
                  />
                  <button
                    type="submit"
                    disabled={!input.trim() || mutual !== true}
                    className="btn-primary p-2.5 disabled:opacity-40"
                  >
                    <Send size={16} />
                  </button>
                </form>
                <p className="text-[10px] text-gray-600 mt-1.5 text-center">
                  Messages are privately analyzed for your wellbeing 💜
                </p>
              </div>
            )}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <MessageCircle size={40} className="text-gray-700 mx-auto mb-3" />
              <p className="text-gray-500 text-sm">Select a conversation or search for someone</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}