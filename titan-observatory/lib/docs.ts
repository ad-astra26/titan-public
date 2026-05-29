import fs from 'fs';
import path from 'path';

export interface DocMeta {
  title: string;
  description: string;
}

export interface DocEntry {
  slug: string;
  meta: DocMeta;
  content: string;
}

export interface SidebarSection {
  label: string;
  items: { label: string; slug: string }[];
}

// Path to the Starlight content directory
const DOCS_ROOT = path.join(process.cwd(), '..', 'titan-docs-site', 'src', 'content', 'docs');

/** Parse frontmatter from markdown file */
function parseFrontmatter(raw: string): { meta: DocMeta; content: string } {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { meta: { title: 'Untitled', description: '' }, content: raw };

  const frontmatter = match[1];
  const content = match[2];

  const title = frontmatter.match(/title:\s*(.+)/)?.[1]?.trim().replace(/^['"]|['"]$/g, '') || 'Untitled';
  const description = frontmatter.match(/description:\s*(.+)/)?.[1]?.trim().replace(/^['"]|['"]$/g, '') || '';

  return { meta: { title, description }, content };
}

/** Load a single doc by slug (e.g., "getting-started/introduction") */
export function getDoc(slug: string): DocEntry | null {
  // Try .md first, then .mdx
  for (const ext of ['.md', '.mdx']) {
    const filePath = path.join(DOCS_ROOT, slug + ext);
    if (fs.existsSync(filePath)) {
      const raw = fs.readFileSync(filePath, 'utf-8');
      const { meta, content } = parseFrontmatter(raw);
      return { slug, meta, content };
    }
  }

  // Try index file in directory
  for (const ext of ['.md', '.mdx']) {
    const filePath = path.join(DOCS_ROOT, slug, 'index' + ext);
    if (fs.existsSync(filePath)) {
      const raw = fs.readFileSync(filePath, 'utf-8');
      const { meta, content } = parseFrontmatter(raw);
      return { slug, meta, content };
    }
  }

  return null;
}

/** Get the landing page doc */
export function getLandingDoc(): DocEntry | null {
  for (const ext of ['.md', '.mdx']) {
    const filePath = path.join(DOCS_ROOT, 'index' + ext);
    if (fs.existsSync(filePath)) {
      const raw = fs.readFileSync(filePath, 'utf-8');
      const { meta, content } = parseFrontmatter(raw);
      return { slug: '', meta, content };
    }
  }
  return null;
}

/** Sidebar navigation structure — only reviewed/approved pages shown */
export const sidebar: SidebarSection[] = [
  {
    label: 'Getting Started',
    items: [
      { label: 'What is Titan?', slug: 'getting-started/introduction' },
    ],
  },
  {
    label: 'Architecture',
    items: [
      { label: 'Sovereignty & Blockchain', slug: 'architecture/sovereignty' },
    ],
  },
  {
    label: 'The Observatory',
    items: [
      { label: 'What is the Observatory?', slug: 'observatory/overview' },
      { label: 'Dashboard Guide', slug: 'observatory/dashboard' },
    ],
  },
  {
    label: 'Setup & Installation',
    items: [
      { label: 'Requirements', slug: 'setup/requirements' },
    ],
  },
  {
    label: 'Contact & Community',
    items: [
      { label: 'Get in Touch', slug: 'contact/get-in-touch' },
      { label: 'Roadmap', slug: 'contact/roadmap' },
    ],
  },
];
