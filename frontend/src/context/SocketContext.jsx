import React, { createContext, useContext, useEffect, useRef, useState } from 'react'
import { useAuth } from './AuthContext'

const SocketContext = createContext(null)

export function SocketProvider({ children }) {
  const { user, api } = useAuth()
  const ws = useRef(null)
  const reconnectTimer = useRef(null)
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState([])
  const listeners = useRef({})

  useEffect(() => {
    if (!user) return

    let disposed = false

    const connect = () => {
      if (disposed) return

      // Build the WS URL from the current origin so it works behind the
      // Vite dev proxy, on a LAN device, or in production — not just
      // localhost:8000.
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const url = `${proto}://${window.location.host}/api/messages/ws/${user.id}`

      ws.current = new WebSocket(url)

      ws.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
      }

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data)

        // Add to messages
        setMessages(prev => [...prev, data])

        // Notify listeners
        if (listeners.current[data.type]) {
          listeners.current[data.type].forEach(cb => cb(data))
        }
      }

      ws.current.onclose = () => {
        setIsConnected(false)
        // Only reconnect while the provider is still mounted and the user
        // is still logged in — otherwise this loops forever on ws/undefined.
        if (!disposed && user) {
          reconnectTimer.current = setTimeout(connect, 3000)
        }
      }

      ws.current.onerror = () => {
        ws.current.close()
      }
    }

    connect()

    return () => {
      disposed = true
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (ws.current) ws.current.close()
    }
  }, [user])

  const sendMessage = (receiverId, content) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        receiver_id: receiverId,
        content,
      }))
    } else {
      // REST fallback so the message still goes through when the socket
      // is down instead of silently dropping it.
      api.post('/messages', {
        sender_id: user.id,
        receiver_id: receiverId,
        content,
      }).catch(() => {})
    }
  }

  const on = (event, callback) => {
    if (!listeners.current[event]) {
      listeners.current[event] = []
    }
    listeners.current[event].push(callback)
    // Return cleanup function
    return () => {
      listeners.current[event] = listeners.current[event].filter(cb => cb !== callback)
    }
  }

  return (
    <SocketContext.Provider value={{
      isConnected,
      messages,
      sendMessage,
      on,
    }}>
      {children}
    </SocketContext.Provider>
  )
}

export const useSocket = () => {
  const ctx = useContext(SocketContext)
  if (!ctx) throw new Error('useSocket must be used within SocketProvider')
  return ctx
}