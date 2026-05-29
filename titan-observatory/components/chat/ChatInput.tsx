'use client';

import { useState, useRef, useCallback, KeyboardEvent, ChangeEvent } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

const MAX_CHARS = 2000;
const MAX_ROWS = 5;

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const lineHeight = 22;
    const maxHeight = lineHeight * MAX_ROWS;
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, []);

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    if (text.length <= MAX_CHARS) {
      setValue(text);
      requestAnimationFrame(adjustHeight);
    }
  };

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [value, disabled, onSend]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const charPct = value.length / MAX_CHARS;

  return (
    <div className="border-t border-titan-metal/10 bg-titan-bg/95 backdrop-blur-sm px-4 py-3">
      <div className="flex items-end gap-3 max-w-4xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={disabled ? 'Titan is thinking...' : 'Message Titan...'}
            rows={1}
            className="w-full resize-none bg-titan-card border border-titan-metal/20 rounded-xl px-4 py-2.5 text-sm text-titan-metal placeholder:text-titan-metal/30 focus:outline-none focus:border-titan-haze/40 focus:ring-1 focus:ring-titan-haze/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ minHeight: '42px' }}
          />
          {value.length > 0 && (
            <span
              className={`absolute bottom-1.5 right-3 text-[10px] ${
                charPct > 0.9 ? 'text-red-400' : 'text-titan-metal/30'
              }`}
            >
              {value.length}/{MAX_CHARS}
            </span>
          )}
        </div>
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="shrink-0 w-10 h-10 rounded-xl bg-titan-haze/90 hover:bg-titan-haze text-titan-bg flex items-center justify-center transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-titan-haze/90"
          title="Send message"
        >
          <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
