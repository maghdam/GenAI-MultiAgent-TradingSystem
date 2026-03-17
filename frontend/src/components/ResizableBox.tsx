import React from 'react';

type ResizeDir = 'e' | 's' | 'se';

type Size = { w?: number; h?: number };

interface Props {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  storageKey?: string;
  defaultWidth?: number;
  defaultHeight?: number;
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  enable?: ResizeDir[];
}

export function ResizableBox({
  children,
  className,
  style,
  storageKey,
  defaultWidth,
  defaultHeight,
  minWidth = 320,
  minHeight = 240,
  maxWidth,
  maxHeight,
  enable = ['e', 's', 'se'],
}: Props) {
  const rootRef = React.useRef<HTMLDivElement | null>(null);
  const dragRef = React.useRef<{
    dir: ResizeDir;
    startX: number;
    startY: number;
    startW: number;
    startH: number;
    maxW: number | undefined;
    maxH: number | undefined;
  } | null>(null);

  const [size, setSize] = React.useState<Size>(() => {
    if (!storageKey) {
      return {
        w: typeof defaultWidth === 'number' ? defaultWidth : undefined,
        h: typeof defaultHeight === 'number' ? defaultHeight : undefined,
      };
    }
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return { w: defaultWidth, h: defaultHeight };
      const parsed = JSON.parse(raw);
      const w = typeof parsed?.w === 'number' ? parsed.w : defaultWidth;
      const h = typeof parsed?.h === 'number' ? parsed.h : defaultHeight;
      return { w, h };
    } catch {
      return { w: defaultWidth, h: defaultHeight };
    }
  });

  const persist = React.useCallback((next: Size) => {
    if (!storageKey) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify({ w: next.w, h: next.h }));
    } catch {
      // ignore
    }
  }, [storageKey]);

  const beginDrag = (dir: ResizeDir, ev: React.PointerEvent<HTMLDivElement>) => {
    ev.preventDefault();
    ev.stopPropagation();

    const root = rootRef.current;
    if (!root) return;

    const rect = root.getBoundingClientRect();
    const parentRect = root.parentElement?.getBoundingClientRect();

    const maxW = maxWidth ?? (parentRect ? Math.floor(parentRect.width) : undefined);
    const maxH = maxHeight;

    dragRef.current = {
      dir,
      startX: ev.clientX,
      startY: ev.clientY,
      startW: rect.width,
      startH: rect.height,
      maxW,
      maxH,
    };

    // If width/height are currently "auto/100%", lock to current px so dragging is stable.
    setSize((prev) => {
      const next = { w: prev.w ?? Math.floor(rect.width), h: prev.h ?? Math.floor(rect.height) };
      persist(next);
      return next;
    });

    try {
      (ev.currentTarget as HTMLDivElement).setPointerCapture(ev.pointerId);
    } catch {
      // ignore
    }
    document.body.style.userSelect = 'none';
    document.body.style.cursor = dir === 'e' ? 'ew-resize' : dir === 's' ? 'ns-resize' : 'nwse-resize';
  };

  React.useEffect(() => {
    const onMove = (ev: PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;

      const dx = ev.clientX - d.startX;
      const dy = ev.clientY - d.startY;

      let nextW = d.startW;
      let nextH = d.startH;

      if (d.dir === 'e' || d.dir === 'se') nextW = d.startW + dx;
      if (d.dir === 's' || d.dir === 'se') nextH = d.startH + dy;

      nextW = Math.max(minWidth, nextW);
      nextH = Math.max(minHeight, nextH);
      if (typeof d.maxW === 'number') nextW = Math.min(d.maxW, nextW);
      if (typeof d.maxH === 'number') nextH = Math.min(d.maxH, nextH);

      const next: Size = { w: Math.floor(nextW), h: Math.floor(nextH) };
      setSize(next);
      persist(next);
    };

    const end = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };

    window.addEventListener('pointermove', onMove, { passive: true });
    window.addEventListener('pointerup', end, { passive: true });
    window.addEventListener('pointercancel', end, { passive: true });
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', end);
      window.removeEventListener('pointercancel', end);
    };
  }, [maxHeight, minHeight, minWidth, persist]);

  const mergedStyle: React.CSSProperties = {
    ...style,
    position: 'relative',
    width: typeof size.w === 'number' ? `${size.w}px` : undefined,
    height: typeof size.h === 'number' ? `${size.h}px` : undefined,
    maxWidth: style?.maxWidth ?? '100%',
  };

  const handleBase: React.CSSProperties = {
    position: 'absolute',
    background: 'transparent',
    touchAction: 'none',
    zIndex: 5,
  };

  return (
    <div ref={rootRef} className={className} style={mergedStyle}>
      {children}

      {enable.includes('e') && (
        <div
          role="separator"
          aria-label="Resize horizontally"
          title="Drag to resize"
          onPointerDown={(e) => beginDrag('e', e)}
          style={{
            ...handleBase,
            top: 0,
            right: 0,
            width: 10,
            height: '100%',
            cursor: 'ew-resize',
          }}
        />
      )}

      {enable.includes('s') && (
        <div
          role="separator"
          aria-label="Resize vertically"
          title="Drag to resize"
          onPointerDown={(e) => beginDrag('s', e)}
          style={{
            ...handleBase,
            left: 0,
            bottom: 0,
            width: '100%',
            height: 10,
            cursor: 'ns-resize',
          }}
        />
      )}

      {enable.includes('se') && (
        <div
          role="separator"
          aria-label="Resize"
          title="Drag to resize"
          onPointerDown={(e) => beginDrag('se', e)}
          style={{
            ...handleBase,
            right: 0,
            bottom: 0,
            width: 18,
            height: 18,
            cursor: 'nwse-resize',
          }}
        />
      )}
    </div>
  );
}

