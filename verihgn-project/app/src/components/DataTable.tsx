import { useMemo, useState, type ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  text?: (row: T) => string;            // used by the search box
  numeric?: boolean;
  width?: string;
}

/** Plain, accessible table with optional text search and paging. */
export function DataTable<T>({ rows, columns, pageSize = 25, searchable = false, searchLabel = "Filter", toolbar,
                               empty = "No rows", caption }: {
  rows: T[]; columns: Column<T>[]; pageSize?: number; searchable?: boolean; searchLabel?: string;
  toolbar?: ReactNode; empty?: string; caption?: string;
}) {
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const filtered = useMemo(() => {
    if (!q.trim()) return rows;
    const needle = q.toLowerCase();
    return rows.filter((r) => columns.some((c) => (c.text ? c.text(r) : "").toLowerCase().includes(needle)));
  }, [rows, columns, q]);
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const p = Math.min(page, pages - 1);
  const shown = filtered.slice(p * pageSize, (p + 1) * pageSize);
  return (
    <div>
      {(searchable || toolbar) && (
        <div className="table-tools">
          {toolbar}
          {searchable && (
            <input className="input" type="search" placeholder={searchLabel} aria-label={searchLabel} value={q}
                   onChange={(e) => { setQ(e.target.value); setPage(0); }} />
          )}
        </div>
      )}
      <div className="table-scroll">
        <table className="table table--stack">
          {caption && <caption className="visually-hidden">{caption}</caption>}
          <thead>
            <tr>{columns.map((c) => <th key={c.key} className={c.numeric ? "num" : undefined} style={c.width ? { width: c.width } : undefined}>{c.header}</th>)}</tr>
          </thead>
          <tbody>
            {shown.length === 0 && <tr><td colSpan={columns.length} className="muted">{empty}</td></tr>}
            {shown.map((r, i) => (
              <tr key={i}>{columns.map((c) => (
                <td key={c.key} className={c.numeric ? "num" : "wrap"} data-label={typeof c.header === "string" ? c.header : c.key}>
                  {c.render(r)}
                </td>
              ))}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {filtered.length > pageSize && (
        <div className="pager">
          <span className="tabular">{p * pageSize + 1}–{Math.min((p + 1) * pageSize, filtered.length)} of {filtered.length}</span>
          <span style={{ display: "flex", gap: 6 }}>
            <button className="icon-button" onClick={() => setPage(Math.max(0, p - 1))} disabled={p === 0}>Previous</button>
            <button className="icon-button" onClick={() => setPage(Math.min(pages - 1, p + 1))} disabled={p >= pages - 1}>Next</button>
          </span>
        </div>
      )}
    </div>
  );
}
