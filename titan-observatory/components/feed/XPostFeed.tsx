'use client';

import { useSocial } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatTimestamp } from '@/lib/formatters';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

export default function XPostFeed() {
  const titanId = useTitanId();
  const { data: social, isLoading } = useSocial(titanId);

  if (isLoading) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-3">
          X Posts
        </h3>
        <LoadingSkeleton lines={4} />
      </div>
    );
  }

  const posts = social?.posts ?? [];

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
          X Posts
        </h3>
        {social?.engagement && (
          <span className="text-[10px] text-titan-metal/40">
            {social.engagement.total_posts} total
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 min-h-0">
        {posts.length === 0 ? (
          <p className="text-xs text-titan-metal/40 text-center py-8">No posts yet</p>
        ) : (
          posts.map((post) => (
            <div
              key={post.id}
              className="bg-titan-bg/40 rounded-lg p-3 border border-titan-metal/5"
            >
              <p className="text-xs text-titan-metal/80 leading-relaxed">
                {post.text}
              </p>
              <div className="flex items-center justify-between mt-2">
                <div className="flex items-center gap-3 text-[10px] text-titan-metal/40">
                  <span>{post.likes} likes</span>
                  <span>{post.replies} replies</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-titan-metal/30">
                    {formatTimestamp(post.timestamp)}
                  </span>
                  {post.url && (
                    <a
                      href={post.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[10px] text-titan-pulse hover:text-titan-pulse/70"
                    >
                      View &#8599;
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
