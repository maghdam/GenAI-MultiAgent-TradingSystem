import React from 'react';

interface CodeDisplayProps {
  code: string;
  language?: string; // reserved for future syntax highlighting
}

export function CodeDisplay({ code }: CodeDisplayProps) {
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      // ignore
    }
  };
  return (
    <div style={{ position: 'relative' }}>
      <button className="btn" type="button" onClick={copy} style={{ position: 'absolute', right: 8, top: 8 }}>
        Copy
      </button>
      <pre style={{ whiteSpace: 'pre-wrap', overflowX: 'auto', margin: 0, paddingTop: 36 }}>{code}</pre>
    </div>
  );
}
