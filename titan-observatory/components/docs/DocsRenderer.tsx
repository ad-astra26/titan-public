'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface DocsRendererProps {
  content: string;
  title: string;
  description: string;
}

export default function DocsRenderer({ content, title, description }: DocsRendererProps) {
  // Strip MDX import/component lines that react-markdown can't handle
  const cleanContent = content
    .replace(/^import\s.*$/gm, '')
    .replace(/<CardGrid[\s\S]*?<\/CardGrid>/g, '')
    .replace(/<Card[\s\S]*?<\/Card>/g, '')
    .replace(/:::(tip|note|caution|danger)\[([^\]]*)\]/g, '> **$2**')
    .replace(/:::/g, '');

  return (
    <article className="max-w-3xl">
      {/* Title and description */}
      <header className="mb-8 pb-6 border-b border-titan-metal/10">
        <h1 className="text-2xl font-bold text-titan-haze mb-2">{title}</h1>
        {description && (
          <p className="text-sm text-titan-metal/60 leading-relaxed">{description}</p>
        )}
      </header>

      {/* Markdown content */}
      <div className="docs-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h2: ({ children }) => (
              <h2 className="text-lg font-semibold text-titan-haze mt-10 mb-4 pb-2 border-b border-titan-metal/10">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-base font-semibold text-titan-haze/90 mt-8 mb-3">
                {children}
              </h3>
            ),
            h4: ({ children }) => (
              <h4 className="text-sm font-semibold text-titan-haze/80 mt-6 mb-2">
                {children}
              </h4>
            ),
            p: ({ children }) => (
              <p className="text-sm text-titan-metal/70 leading-relaxed mb-4">
                {children}
              </p>
            ),
            ul: ({ children }) => (
              <ul className="list-disc list-outside pl-5 space-y-1.5 mb-4 text-sm text-titan-metal/70">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="list-decimal list-outside pl-5 space-y-1.5 mb-4 text-sm text-titan-metal/70">
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li className="leading-relaxed">{children}</li>
            ),
            strong: ({ children }) => (
              <strong className="text-titan-haze/90 font-semibold">{children}</strong>
            ),
            a: ({ href, children }) => (
              <a
                href={href}
                className="text-titan-haze underline underline-offset-2 decoration-titan-haze/30 hover:decoration-titan-haze transition-colors"
              >
                {children}
              </a>
            ),
            code: ({ className, children }) => {
              const isInline = !className;
              if (isInline) {
                return (
                  <code className="text-xs bg-titan-card/80 text-titan-growth px-1.5 py-0.5 rounded font-mono border border-titan-metal/10">
                    {children}
                  </code>
                );
              }
              return (
                <code className={className}>{children}</code>
              );
            },
            pre: ({ children }) => (
              <pre className="bg-[#07090d] border border-titan-metal/15 rounded-xl p-4 overflow-x-auto mb-4 text-xs font-mono leading-relaxed">
                {children}
              </pre>
            ),
            table: ({ children }) => (
              <div className="overflow-x-auto mb-4 rounded-xl border border-titan-metal/10">
                <table className="w-full text-xs">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-titan-card/80 text-titan-haze">
                {children}
              </thead>
            ),
            th: ({ children }) => (
              <th className="text-left px-3 py-2.5 font-semibold border-b border-titan-metal/15">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-3 py-2 text-titan-metal/70 border-b border-titan-metal/8">
                {children}
              </td>
            ),
            blockquote: ({ children }) => (
              <blockquote className="border-l-2 border-titan-haze/40 pl-4 my-4 py-2 bg-titan-haze/5 rounded-r-lg">
                {children}
              </blockquote>
            ),
            hr: () => (
              <hr className="border-titan-metal/10 my-8" />
            ),
          }}
        >
          {cleanContent}
        </ReactMarkdown>
      </div>
    </article>
  );
}
