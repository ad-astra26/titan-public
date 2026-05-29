'use client';

import ChatWindow from '@/components/chat/ChatWindow';
import TitanSelector from '@/components/shared/TitanSelector';

export default function ChatPage() {
  return (
    <div className="flex flex-col h-full">
      <TitanSelector />
      <ChatWindow />
    </div>
  );
}
