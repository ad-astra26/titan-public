'use client';

import React from 'react';

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="bg-titan-card rounded-xl p-6 text-center">
          <p className="text-red-400 text-sm mb-1">Component Error</p>
          <p className="text-titan-metal/40 text-xs font-mono">
            {this.state.error?.message ?? 'Unknown error'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-3 text-xs text-titan-haze hover:text-titan-haze/80 underline"
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
